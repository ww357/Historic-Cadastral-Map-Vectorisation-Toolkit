"""
Pre-train SAM DoRA weights for parcel segmentation from labelme annotations.

This is step 03 in the parcel pipeline — run it BEFORE 04_predict to give SAM
domain knowledge of tithe-map parcel boundaries from a small set of hand-drawn
annotation examples.

Differs from MapSAM polygon fine-tuning (steps/03_finetune/polygons/train.py) in
two important ways:
  1. Trained WITH point prompts — the foreground centroid + simulated background
     points are included in every forward pass, matching the inference setup in
     parcel_predict.py.  This teaches SAM to respond correctly when asked to
     "segment the region at this point, not those adjacent ones."
  2. Multi-mask candidate selection — SAM's three mask candidates are evaluated
     against the GT mask; only the best-matching one receives the loss.  This
     trains SAM to put its best answer in the top-scoring slot (which inference
     selects with argmax(scores)).

Training data comes from annotations made with labelme on 1024px patches
produced by parcel_patchify.py, then exported with export_masks.py.
After training the weights are auto-detected by parcel_predict.py.

Full workflow
-------------
    # Step 01 — create 1024px patches for annotation
    python steps/01_patchify/parcels/parcel_patchify.py --sheet Timberscombe

    # Step 02 — annotate with labelme (draw "parcel" polygons)
    labelme data/patches/parcel/Timberscombe/

    # Step 02 — export binary masks from labelme JSON
    python steps/02_annotate/export_masks.py --sheet Timberscombe \\
        --patches-dir data/patches/parcel/Timberscombe \\
        --json-dir    data/patches/parcel/Timberscombe

    # Step 03 — fine-tune (this script)
    conda activate polygons
    python steps/03_finetune/parcels/train.py --sheet Timberscombe
    # or pool multiple sheets:
    python steps/03_finetune/parcels/train.py --sheet Timberscombe Dunster
    # or use everything annotated so far:
    python steps/03_finetune/parcels/train.py --all-sheets

    # Step 04 — predict (picks up DoRA weights automatically)
    python steps/04_predict/parcels/parcel_predict.py --sheet Timberscombe

Outputs
-------
    models/finetuned/sam_parcels_<name>_best.pth
    models/logs/sam_parcels_<name>_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT       = Path(__file__).resolve().parents[3]
MAPSAM_DIR = ROOT / "models" / "MapSAM"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MAPSAM_DIR))

from sam_dora_image_encoder import DoRA_Sam      # noqa: E402
from segment_anything import sam_model_registry  # noqa: E402


# ── Point prompt simulation ───────────────────────────────────────────────────

def _sample_fg_point(mask: np.ndarray) -> tuple[float, float] | None:
    """
    Return (col, row) of a foreground point guaranteed to lie on the mask,
    or None if the mask is empty.

    The centroid of an irregular or concave parcel can fall outside the mask.
    When that happens the nearest foreground pixel to the centroid is used as
    the base point before jitter is applied, so the prompt always overlaps the
    annotated region.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    cx, cy = float(xs.mean()), float(ys.mean())

    # Snap centroid to nearest foreground pixel if it falls outside the mask
    # (common for concave or L/U-shaped parcels).
    if mask[int(round(cy)), int(round(cx))] == 0:
        dists   = (xs - cx) ** 2 + (ys - cy) ** 2
        nearest = int(np.argmin(dists))
        cx, cy  = float(xs[nearest]), float(ys[nearest])

    jitter = mask.shape[0] * 0.04
    cx2 = float(np.clip(cx + np.random.uniform(-jitter, jitter), 0, mask.shape[1]-1))
    cy2 = float(np.clip(cy + np.random.uniform(-jitter, jitter), 0, mask.shape[0]-1))
    if mask[int(cy2), int(cx2)] == 0:
        cx2, cy2 = cx, cy
    return cx2, cy2


def _sample_bg_points(mask: np.ndarray, n: int,
                       min_dist: int = 8) -> np.ndarray:
    """
    Sample n background (col, row) points well outside the mask.
    Uses a dilated exclusion zone around the mask to keep points in clear
    background, simulating the neighbouring-parcel negative prompts used in
    parcel_predict.py.
    """
    h, w    = mask.shape
    dilated = cv2.dilate(mask,
                         np.ones((min_dist*2+1, min_dist*2+1), np.uint8),
                         iterations=1)
    bg_ys, bg_xs = np.where(dilated == 0)
    if len(bg_xs) == 0:
        return np.array([[5.0, 5.0], [w-5.0, 5.0],
                          [5.0, h-5.0], [w-5.0, h-5.0]][:n])
    n   = min(n, len(bg_xs))
    idx = np.random.choice(len(bg_xs), n, replace=False)
    return np.column_stack([bg_xs[idx].astype(float),
                             bg_ys[idx].astype(float)])


# ── Augmentation ──────────────────────────────────────────────────────────────

def _augment(img: np.ndarray, mask: np.ndarray,
             min_zoom_area_px: int = 1500) -> tuple[np.ndarray, np.ndarray]:
    """
    Augmentation for parcel SAM training.

    Three independent transforms applied sequentially:

    1. rot90 + flip  (50%)         — orientation invariance
    2. Small rotation ±15°  (25%)  — slight perspective variation
    3. Random zoom-out  (40%)      — MULTI-SCALE: simulates the adaptive 2×
       scale path in parcel_predict.py where a 2048px window is downsampled
       to 1024px.  A centre-crop of 1.2–2.0× is resized back to the original
       size, shrinking boundaries proportionally.  Without this the model only
       ever sees 0.5m/px resolution during training, but the 2× path shows it
       boundaries at 1m/px — half the pixel density.

       Skipped if the parcel mask area is below min_zoom_area_px: very small
       parcels become indistinguishable after a 2× downscale so zooming out
       would produce a misleading training signal rather than a useful one.
       At 0.5m/px native resolution, 1500 px ≈ 375 m² — at 2× zoom that
       leaves ~375 px, roughly a 19×19-pixel blob, still learnable.
    """
    h, w = img.shape[:2]

    # 1. rot90 + flip
    if random.random() > 0.5:
        k    = np.random.randint(0, 4)
        img  = np.rot90(img,  k).copy()
        mask = np.rot90(mask, k).copy()
        if random.random() > 0.5:
            img  = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        elif random.random() > 0.5:
            img  = np.flipud(img).copy()
            mask = np.flipud(mask).copy()

    # 2. Small rotation
    if random.random() > 0.75:
        angle = np.random.uniform(-15, 15)
        M     = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img   = cv2.warpAffine(img,  M, (w, h),
                                borderMode=cv2.BORDER_REFLECT_101)
        mask  = cv2.warpAffine(mask, M, (w, h),
                                flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)

    # 3. Multi-scale zoom-out (simulates 2× adaptive scale path)
    mask_area = int((mask > 127).sum())
    if random.random() > 0.6 and mask_area >= min_zoom_area_px:
        factor  = np.random.uniform(1.2, 2.0)   # zoom out 1.2× – 2×
        crop_h  = int(h / factor)
        crop_w  = int(w / factor)
        # Centre-crop so the annotated parcel stays in frame
        sy = (h - crop_h) // 2
        sx = (w - crop_w) // 2
        img_c  = img [sy:sy+crop_h, sx:sx+crop_w]
        mask_c = mask[sy:sy+crop_h, sx:sx+crop_w]
        # Resize back to original dimensions — boundary density halves at 2×
        img    = cv2.resize(img_c,  (w, h), interpolation=cv2.INTER_AREA)
        mask   = cv2.resize(mask_c, (w, h), interpolation=cv2.INTER_NEAREST)

    return img, mask


# ── Dataset ───────────────────────────────────────────────────────────────────

class ParcelAnnotationDataset(Dataset):
    """
    Loads (image, mask) pairs from the standard annotation export format:
        data/annotations/parcel/<sheet>/images/*.png
        data/annotations/parcel/<sheet>/masks/*.png

    Each item includes simulated point prompts derived from the GT mask so the
    model trains in the same conditions as parcel_predict.py inference.
    """

    def __init__(self, pairs: list[tuple[Path, Path]],
                 img_size: int, n_bg: int = 6, do_aug: bool = True,
                 min_zoom_area_px: int = 1500):
        # Filter out pairs whose mask has no foreground pixels — an empty mask
        # produces a point prompt that cannot overlap the annotation, which would
        # corrupt the training signal.
        valid = []
        for img_p, mask_p in pairs:
            m = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            if m is not None and (m > 0).any():
                valid.append((img_p, mask_p))
            else:
                print(f"  WARNING: empty mask skipped — {mask_p.name}")
        self.pairs            = valid
        self.img_size         = img_size
        self.n_bg             = n_bg
        self.do_aug           = do_aug
        self.min_zoom_area_px = min_zoom_area_px

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path = self.pairs[idx]

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(img_path)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)

        if self.do_aug:
            img, mask = _augment(img, mask, self.min_zoom_area_px)

        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img  = cv2.resize(img,  (self.img_size, self.img_size),
                              interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size),
                              interpolation=cv2.INTER_NEAREST)

        # ── Point prompts ─────────────────────────────────────────────────────
        fg = _sample_fg_point(mask)
        if fg is None:
            # Empty mask slipped through — __init__ filter should have caught it
            raise ValueError(f"Empty mask in training data: {mask_path}")
        bg = _sample_bg_points(mask, self.n_bg)

        coords = np.vstack([[list(fg)], bg.tolist()]).astype(np.float32)
        labels = np.array([1] + [0]*len(bg), dtype=np.int32)

        # GT mask at SAM's internal low-res output size (img_size // 4)
        low_sz   = self.img_size // 4
        mask_bin = (mask > 127).astype(np.float32)
        low_mask = cv2.resize(mask_bin, (low_sz, low_sz),
                              interpolation=cv2.INTER_NEAREST)

        img_t = torch.from_numpy(
            np.ascontiguousarray(img, dtype=np.uint8)
        ).float() / 255.0
        img_t = img_t.permute(2, 0, 1)   # (3, H, W)

        return {
            "image"    : img_t,
            "mask_low" : torch.from_numpy(low_mask).float(),
            "coords"   : torch.from_numpy(coords),
            "labels"   : torch.from_numpy(labels),
            "case_name": img_path.stem,
        }


# ── Loss ──────────────────────────────────────────────────────────────────────

def _dice(logits: torch.Tensor, target: torch.Tensor,
          eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(logits).flatten(1)
    t = target.float().flatten(1)
    return (1. - (2. * (p*t).sum(1) + eps) /
            (p.sum(1) + t.sum(1) + eps)).mean()


def _focal(logits: torch.Tensor, target: torch.Tensor,
           alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    i = logits.flatten(1)
    t = target.float().flatten(1)
    ce = F.binary_cross_entropy_with_logits(i, t, reduction="none")
    pt = i.sigmoid() * t + (1 - i.sigmoid()) * (1 - t)
    return ((alpha*t + (1-alpha)*(1-t)) * ce * (1-pt)**gamma).mean(1).mean()


def combined_loss(logits: torch.Tensor, target: torch.Tensor,
                  dice_w: float = 0.8) -> torch.Tensor:
    return (1-dice_w)*_focal(logits, target) + dice_w*_dice(logits, target)


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def _validate(model, loader: DataLoader, img_size: int,
              device: torch.device) -> float:
    model.eval()
    iou_sum, n = 0.0, 0
    low_sz = img_size // 4
    for batch in loader:
        imgs   = batch["image"].to(device)
        coords = batch["coords"].to(device)
        labels = batch["labels"].to(device)
        gt_low = batch["mask_low"].to(device)

        imgs_pre = model.sam.preprocess(imgs * 255.0)
        img_emb  = model.sam.image_encoder(imgs_pre)
        if isinstance(img_emb, tuple):
            img_emb = img_emb[0]

        for b in range(imgs.shape[0]):
            sp, dp = model.sam.prompt_encoder(
                points=(coords[b:b+1], labels[b:b+1]),
                boxes=None, masks=None,
            )
            lr, iou_p = model.sam.mask_decoder(
                image_embeddings  = img_emb[b:b+1],
                image_pe          = model.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings = sp,
                dense_prompt_embeddings  = dp,
                multimask_output  = True,
            )
            best = int(iou_p[0].argmax())
            lr_up = F.interpolate(lr[:, best:best+1],
                                   size=(low_sz, low_sz),
                                   mode="bilinear", align_corners=False)
            pred  = (torch.sigmoid(lr_up[0, 0]) > 0.5).float()
            gt    = gt_low[b]
            inter = (pred * gt).sum()
            union = pred.sum() + gt.sum() - inter
            iou_sum += ((inter + 1e-6) / (union + 1e-6)).item()
            n += 1

    model.train()
    return iou_sum / n if n else 0.0


# ── Weight resolution ─────────────────────────────────────────────────────────

def _resolve_weights(explicit: str | None, finetuned_dir: Path,
                     base_dir: Path) -> str:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            return str(p)
        sys.exit(f"Weights not found: {p}")

    candidates = sorted(finetuned_dir.glob("sam_parcels*_best.pth"),
                        key=lambda p: p.stat().st_mtime)
    if candidates:
        print(f"Resuming from: {candidates[-1].name}")
        return str(candidates[-1])

    fallback = base_dir / "origional_weights" / "sam_vit_b_01ec64.pth"
    if fallback.exists():
        print("No previous parcel weights — initialising DoRA from SAM base.")
        return str(fallback)

    sys.exit(f"SAM weights not found at {fallback}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-train SAM DoRA for parcel segmentation from labelme annotations."
    )
    sheet_grp = parser.add_mutually_exclusive_group(required=True)
    sheet_grp.add_argument("--sheet", nargs="+", metavar="SHEET",
                           help="One or more sheet IDs to pool for training, e.g. "
                                "--sheet Timberscombe Dunster")
    sheet_grp.add_argument("--all-sheets", action="store_true",
                           help="Use all sheets found in data/annotations/parcel/")
    parser.add_argument("--name",    default=None,
                        help="Run name (auto-generated if omitted)")
    parser.add_argument("--weights", default=None,
                        help="Explicit .pth file to start from")
    parser.add_argument("--config",  default="config.yaml")
    args = parser.parse_args()

    cfg    = yaml.safe_load((ROOT / args.config).read_text())
    paths  = cfg["paths"]
    mcfg   = cfg["mapsam"]
    ft_cfg = cfg.get("parcel_finetune", {})
    pcfg   = cfg.get("parcels", {})

    img_size  = int(pcfg.get("sam_input_size", 1024))
    rank      = int(mcfg.get("rank", 4))
    dice_w    = float(mcfg.get("dice_param", 0.8))
    epochs    = int(ft_cfg.get("epochs", 40))
    bs        = int(ft_cfg.get("batch_size", 1))
    lr        = float(ft_cfg.get("learning_rate", 5e-5))
    val_split = float(ft_cfg.get("val_split", 0.15))
    patience  = int(ft_cfg.get("early_stopping_patience", 12))
    n_bg             = int(pcfg.get("max_neg_points", 6))
    min_zoom_area_px = int(ft_cfg.get("min_zoom_area_px", 1500))
    seed             = int(ft_cfg.get("seed", 1234))

    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device   : {device}")

    # ── Resolve sheet list ────────────────────────────────────────────────────
    feature  = "parcel"
    base_ann = ROOT / paths["annotations"] / feature

    if args.all_sheets:
        sheet_ids = sorted(d.name for d in base_ann.iterdir() if d.is_dir()) \
                    if base_ann.exists() else []
        if not sheet_ids:
            sys.exit(f"No annotation directories found under {base_ann}")
        print(f"Sheets   : all ({len(sheet_ids)}) — {', '.join(sheet_ids)}")
    else:
        sheet_ids = args.sheet
        print(f"Sheets   : {', '.join(sheet_ids)}")

    # ── Auto-export masks for any sheet that only has labelme JSONs ───────────
    for sheet in sheet_ids:
        imgs_dir = base_ann / sheet / "images"
        if not imgs_dir.exists() or not any(imgs_dir.glob("*.png")):
            patches_dir = ROOT / "data" / "patches" / "parcel" / sheet
            if patches_dir.exists() and any(patches_dir.glob("*.json")):
                print(f"  [{sheet}] masks not found — running export_masks.py ...")
                result = subprocess.run(
                    [sys.executable,
                     str(ROOT / "steps" / "02_annotate" / "export_masks.py"),
                     "--sheet",       sheet,
                     "--patches-dir", str(patches_dir),
                     "--json-dir",    str(patches_dir)],
                    check=False,
                )
                if result.returncode != 0:
                    sys.exit(f"export_masks.py failed for sheet {sheet}.")
            else:
                sys.exit(
                    f"No training data found for sheet '{sheet}'.\n"
                    f"  Expected: {imgs_dir}\n"
                    f"  Or labelme JSONs at: {patches_dir}\n\n"
                    f"Run parcel_patchify.py → annotate_parcels.py → export_masks.py first."
                )

    finetuned_dir = ROOT / paths["models_finetuned"]
    logs_dir      = ROOT / paths["logs"]
    base_dir      = ROOT / paths["models_base"] / "MapSAM"
    finetuned_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    sam_ckpt = base_dir / "origional_weights" / "sam_vit_b_01ec64.pth"
    if not sam_ckpt.exists():
        sys.exit(f"SAM base weights not found: {sam_ckpt}")

    sam, _ = sam_model_registry["vit_b"](
        image_size  = img_size,
        num_classes = 4,
        checkpoint  = str(sam_ckpt),
        pixel_mean  = [0, 0, 0],
        pixel_std   = [1, 1, 1],
    )
    model = DoRA_Sam(sam, rank).to(device)

    weights_path = _resolve_weights(args.weights, finetuned_dir, base_dir)
    if not weights_path.endswith("sam_vit_b_01ec64.pth"):
        model.load_dora_parameters(weights_path)
        print(f"Weights  : {Path(weights_path).name}  (DoRA resumed)")
    else:
        print(f"Weights  : {Path(weights_path).name}  (SAM base — fresh DoRA)")

    # ── Data — pool pairs from all sheets ─────────────────────────────────────
    all_pairs = []
    for sheet in sheet_ids:
        imgs_dir = base_ann / sheet / "images"
        msks_dir = base_ann / sheet / "masks"
        sheet_pairs = sorted(
            [(imgs_dir / p.name, msks_dir / p.name)
             for p in imgs_dir.glob("*.png")
             if (msks_dir / p.name).exists()]
        )
        print(f"  [{sheet}] {len(sheet_pairs)} annotated parcels")
        all_pairs.extend(sheet_pairs)
    if not all_pairs:
        sys.exit(f"No image+mask pairs found in {ann_dir}")

    print(f"Patches  : {len(all_pairs)} annotated parcels")
    if len(all_pairs) < 10:
        print("  ⚠  Fewer than 10 examples — fine-tuning will be noisy. "
              "Annotate more parcels for better results.")

    rng  = np.random.default_rng(seed)
    perm = rng.permutation(len(all_pairs))
    n_val = max(1, int(len(all_pairs) * val_split))

    val_set     = {all_pairs[i][0].stem for i in perm[:n_val]}
    train_pairs = [p for p in all_pairs if p[0].stem not in val_set]
    val_pairs   = [p for p in all_pairs if p[0].stem in val_set]
    print(f"Split    : {len(train_pairs)} train / {len(val_pairs)} val")

    train_ds = ParcelAnnotationDataset(train_pairs, img_size, n_bg=n_bg, do_aug=True,
                                       min_zoom_area_px=min_zoom_area_px)
    val_ds   = ParcelAnnotationDataset(val_pairs,   img_size, n_bg=n_bg, do_aug=False)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,  shuffle=False,
                              num_workers=0, pin_memory=True)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, betas=(0.9, 0.999), weight_decay=0.01,
    )
    max_iters = epochs * max(len(train_loader), 1)

    def _cosine_lr(i: int) -> float:
        return lr * 0.5 * (1.0 + math.cos(math.pi * i / max(max_iters, 1)))

    # ── Run outputs ───────────────────────────────────────────────────────────
    run_name  = args.name or datetime.now().strftime("%Y%m%d_%H%M")
    run_name  = f"sam_parcels_{run_name}"
    best_path = finetuned_dir / f"{run_name}_best.pth"
    log_path  = logs_dir      / f"{run_name}_metrics.csv"

    print(f"\nRun      : {run_name}")
    print(f"Best  →    {best_path.relative_to(ROOT)}")
    print(f"Log   →    {log_path.relative_to(ROOT)}\n")

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_iou"])

    best_iou   = -1.0
    no_improve = 0
    iter_num   = 0
    low_sz     = img_size // 4

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    for epoch in tqdm(range(epochs), desc="Epochs", ncols=70):
        epoch_loss = 0.0

        for batch in train_loader:
            imgs   = batch["image"].to(device)
            coords = batch["coords"].to(device)
            labels = batch["labels"].to(device)
            gt_low = batch["mask_low"].to(device)

            # Image encoding — DoRA adapters are active here (trainable)
            imgs_pre = model.sam.preprocess(imgs * 255.0)
            img_emb  = model.sam.image_encoder(imgs_pre)
            if isinstance(img_emb, tuple):
                img_emb = img_emb[0]

            total_loss = torch.tensor(0.0, device=device)

            for b in range(imgs.shape[0]):
                # Prompt + mask decoder are frozen — no gradient flows through them
                with torch.no_grad():
                    sp, dp = model.sam.prompt_encoder(
                        points=(coords[b:b+1], labels[b:b+1]),
                        boxes=None, masks=None,
                    )

                lr_masks, iou_preds = model.sam.mask_decoder(
                    image_embeddings         = img_emb[b:b+1],
                    image_pe                 = model.sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings = sp,
                    dense_prompt_embeddings  = dp,
                    multimask_output         = True,
                )
                # Upsample to match GT low-res size
                lr_up  = F.interpolate(lr_masks,
                                        size=(low_sz, low_sz),
                                        mode="bilinear", align_corners=False)
                gt_exp = gt_low[b:b+1].unsqueeze(1)

                # Select best mask candidate by IoU with GT
                with torch.no_grad():
                    pbin = (torch.sigmoid(lr_up) > 0.5).float()
                    ious = []
                    for mi in range(lr_up.shape[1]):
                        p  = pbin[:, mi]
                        t  = gt_exp[:, 0]
                        i_ = (p * t).sum()
                        u_ = p.sum() + t.sum() - i_
                        ious.append(((i_ + 1e-6) / (u_ + 1e-6)).item())
                    best = int(np.argmax(ious))

                # Loss only on the best-matching candidate
                total_loss = total_loss + combined_loss(
                    lr_up[:, best:best+1], gt_exp, dice_w
                )

            total_loss = total_loss / imgs.shape[0]
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            for pg in optimizer.param_groups:
                pg["lr"] = _cosine_lr(iter_num)
            iter_num   += 1
            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        val_iou  = _validate(model, val_loader, img_size, device)

        print(f"  Epoch {epoch+1:3d}/{epochs}  "
              f"loss={avg_loss:.4f}  val_IoU={val_iou:.4f}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, round(avg_loss, 6), round(val_iou, 4)])

        if val_iou > best_iou:
            best_iou   = val_iou
            no_improve = 0
            model.save_dora_parameters(str(best_path))
            print(f"    ✓ new best val_IoU={val_iou:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping — best val_IoU={best_iou:.4f}")
                break

    print(f"\nDone. Best val_IoU={best_iou:.4f}")
    print(f"  Weights : {best_path.relative_to(ROOT)}")
    print(f"  Log     : {log_path.relative_to(ROOT)}")
    print(f"\n  parcel_predict.py will pick these weights up automatically.")


if __name__ == "__main__":
    main()
