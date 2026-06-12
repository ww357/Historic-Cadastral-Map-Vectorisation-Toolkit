"""
Feedback fine-tune SAM DoRA weights for parcel segmentation from corrected QGIS predictions.

Differs from 03_finetune/parcels/train.py in three ways:

  1. Data source: training pairs come from prepare_parcels.py output
     (data/annotations/parcels/<sheet>/) — i.e. rasterised QGIS-corrected polygons —
     rather than hand-drawn labelme annotations.

  2. Replay buffer: pre-training annotation patches from ALL sheets
     (data/annotations/parcel/<any sheet>/) are mixed into every training epoch
     at replay_ratio.  This prevents the DoRA adapters from drifting away from
     the multi-sheet knowledge built during 03_finetune.

  3. Cross-sheet validation: the validation set is drawn from the replay pool
     (annotations from other sheets), not a split of the current feedback data.
     Early stopping triggers when cross-sheet IoU stops improving, which catches
     overfitting to one sheet before the replay data loses its influence.

Fallback: if no pre-training annotation data exists yet, the script falls back to
splitting the feedback data 85/15 (original behaviour) with a warning.

Workflow:
    conda activate polygons
    python steps/06_feedback/parcels/prepare_parcels.py --sheet Timberscombe
    python steps/06_feedback/parcels/train.py --sheet Timberscombe

Output:
    models/finetuned/sam_parcels_fb_<name>_best.pth
    models/logs/sam_parcels_fb_<name>_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
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


# ── Point prompt helpers ──────────────────────────────────────────────────────

def _sample_fg_point(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    if mask[int(round(cy)), int(round(cx))] == 0:
        dists   = (xs - cx) ** 2 + (ys - cy) ** 2
        nearest = int(np.argmin(dists))
        cx, cy  = float(xs[nearest]), float(ys[nearest])
    jitter = mask.shape[0] * 0.05
    cx2 = float(np.clip(cx + np.random.uniform(-jitter, jitter), 0, mask.shape[1] - 1))
    cy2 = float(np.clip(cy + np.random.uniform(-jitter, jitter), 0, mask.shape[0] - 1))
    if mask[int(cy2), int(cx2)] == 0:
        cx2, cy2 = cx, cy
    return cx2, cy2


def _sample_bg_points(mask: np.ndarray, n: int, min_dist: int = 10) -> np.ndarray:
    h, w    = mask.shape
    dilated = cv2.dilate(mask,
                         np.ones((min_dist * 2 + 1, min_dist * 2 + 1), np.uint8),
                         iterations=1)
    bg_ys, bg_xs = np.where(dilated == 0)
    if len(bg_xs) == 0:
        return np.array([[5.0, 5.0], [w - 5.0, 5.0],
                          [5.0, h - 5.0], [w - 5.0, h - 5.0]][:n])
    n   = min(n, len(bg_xs))
    idx = np.random.choice(len(bg_xs), n, replace=False)
    return np.column_stack([bg_xs[idx].astype(float), bg_ys[idx].astype(float)])


# ── Augmentation ──────────────────────────────────────────────────────────────

def _augment(img: np.ndarray, mask: np.ndarray,
             min_zoom_area_px: int = 1500) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    if random.random() > 0.5:
        k    = np.random.randint(0, 4)
        img  = np.rot90(img,  k).copy()
        mask = np.rot90(mask, k).copy()
        if random.random() > 0.5:
            img  = np.fliplr(img).copy();  mask = np.fliplr(mask).copy()
        elif random.random() > 0.5:
            img  = np.flipud(img).copy();  mask = np.flipud(mask).copy()
    if random.random() > 0.75:
        angle = np.random.uniform(-15, 15)
        M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img   = cv2.warpAffine(img,  M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        mask  = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_area = int((mask > 127).sum())
    if random.random() > 0.6 and mask_area >= min_zoom_area_px:
        factor = np.random.uniform(1.2, 2.0)
        ch, cw = int(h / factor), int(w / factor)
        sy, sx = (h - ch) // 2, (w - cw) // 2
        img    = cv2.resize(img [sy:sy+ch, sx:sx+cw], (w, h), interpolation=cv2.INTER_AREA)
        mask   = cv2.resize(mask[sy:sy+ch, sx:sx+cw], (w, h), interpolation=cv2.INTER_NEAREST)
    return img, mask


# ── Dataset ───────────────────────────────────────────────────────────────────

class ParcelDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]],
                 img_size: int, n_bg: int = 6,
                 do_aug: bool = True, min_zoom_area_px: int = 1500):
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
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        if self.do_aug:
            img, mask = _augment(img, mask, self.min_zoom_area_px)
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img  = cv2.resize(img,  (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        fg = _sample_fg_point(mask)
        if fg is None:
            raise ValueError(f"Empty mask in training data: {mask_path}")
        bg     = _sample_bg_points(mask, self.n_bg)
        coords = np.vstack([[list(fg)], bg.tolist()]).astype(np.float32)
        labels = np.array([1] + [0] * len(bg), dtype=np.int32)
        low_sz   = self.img_size // 4
        mask_bin = (mask > 127).astype(np.float32)
        low_mask = cv2.resize(mask_bin, (low_sz, low_sz), interpolation=cv2.INTER_NEAREST)
        img_t = torch.from_numpy(
            np.ascontiguousarray(img, dtype=np.uint8)).float() / 255.0
        img_t = img_t.permute(2, 0, 1)
        return {
            "image"    : img_t,
            "mask_low" : torch.from_numpy(low_mask).float(),
            "coords"   : torch.from_numpy(coords),
            "labels"   : torch.from_numpy(labels),
            "case_name": img_path.stem,
        }


# ── Loss ──────────────────────────────────────────────────────────────────────

def _dice(logits, target, eps=1e-6):
    p = torch.sigmoid(logits).flatten(1)
    t = target.float().flatten(1)
    return (1.0 - (2.0 * (p * t).sum(1) + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def _focal(logits, target, alpha=0.25, gamma=2.0):
    inp = logits.flatten(1)
    tgt = target.float().flatten(1)
    ce  = F.binary_cross_entropy_with_logits(inp, tgt, reduction="none")
    pt  = inp.sigmoid() * tgt + (1 - inp.sigmoid()) * (1 - tgt)
    return ((alpha * tgt + (1 - alpha) * (1 - tgt)) * ce * (1 - pt) ** gamma).mean(1).mean()


def combined_loss(logits, target, dice_w=0.8):
    return (1 - dice_w) * _focal(logits, target) + dice_w * _dice(logits, target)


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader: DataLoader, img_size: int, device: torch.device) -> float:
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
                points=(coords[b:b+1], labels[b:b+1]), boxes=None, masks=None)
            lr_masks, iou_preds = model.sam.mask_decoder(
                image_embeddings=img_emb[b:b+1],
                image_pe=model.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sp,
                dense_prompt_embeddings=dp,
                multimask_output=True,
            )
            best = int(iou_preds[0].argmax())
            lr   = F.interpolate(lr_masks[:, best:best+1],
                                  size=(low_sz, low_sz),
                                  mode="bilinear", align_corners=False)
            pred  = (torch.sigmoid(lr[0, 0]) > 0.5).float()
            gt    = gt_low[b]
            inter = (pred * gt).sum()
            union = pred.sum() + gt.sum() - inter
            iou_sum += ((inter + 1e-6) / (union + 1e-6)).item()
            n += 1
    model.train()
    return iou_sum / n if n else 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_pairs(ann_dir: Path) -> list[tuple[Path, Path]]:
    """Collect (image, mask) pairs from a standard annotations directory."""
    imgs_dir = ann_dir / "images"
    msks_dir = ann_dir / "masks"
    if not imgs_dir.exists():
        return []
    return sorted(
        [(imgs_dir / p.name, msks_dir / p.name)
         for p in imgs_dir.glob("*.png")
         if (msks_dir / p.name).exists()]
    )


def _resolve_weights(explicit: str | None, finetuned_dir: Path,
                     base_dir: Path) -> str:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            return str(p)
        sys.exit(f"Weights not found: {p}")
    # Prefer most-recent feedback weights, then any parcel weights, then SAM base
    for pattern in ("sam_parcels_fb*_best.pth", "sam_parcels*_best.pth"):
        candidates = sorted(finetuned_dir.glob(pattern),
                            key=lambda p: p.stat().st_mtime)
        if candidates:
            return str(candidates[-1])
    fallback = base_dir / "origional_weights" / "sam_vit_b_01ec64.pth"
    if fallback.exists():
        print("  No parcel weights found — initialising DoRA from SAM base.")
        return str(fallback)
    sys.exit(f"No SAM weights found.\n  Searched: {finetuned_dir}\n  Fallback: {fallback}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feedback fine-tune SAM DoRA from QGIS-corrected parcel predictions."
    )
    parser.add_argument("--sheet",   required=True,
                        help="Sheet ID — feedback data must exist in "
                             "data/annotations/parcels/<sheet>/")
    parser.add_argument("--name",    default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--config",  default="config.yaml")
    args = parser.parse_args()

    cfg    = yaml.safe_load((ROOT / args.config).read_text())
    paths  = cfg["paths"]
    mcfg   = cfg["mapsam"]
    ft_cfg = cfg.get("parcel_finetune", {})
    pcfg   = cfg.get("parcels", {})

    img_size         = int(pcfg.get("sam_input_size", 1024))
    rank             = int(mcfg.get("rank", 4))
    dice_w           = float(mcfg.get("dice_param", 0.8))
    epochs           = int(ft_cfg.get("epochs", 30))
    bs               = int(ft_cfg.get("batch_size", 1))
    lr               = float(ft_cfg.get("learning_rate", 5e-5))
    val_split        = float(ft_cfg.get("val_split", 0.15))
    patience         = int(ft_cfg.get("early_stopping_patience", 10))
    n_bg             = int(pcfg.get("max_neg_points", 6))
    min_zoom_area_px = int(ft_cfg.get("min_zoom_area_px", 1500))
    replay_ratio     = float(ft_cfg.get("replay_ratio", 0.4))
    seed             = int(ft_cfg.get("seed", 1234))

    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device    : {device}")

    # ── Feedback data (QGIS-corrected predictions for this sheet) ─────────────
    fb_ann_dir = ROOT / paths["annotations"] / "parcels" / args.sheet
    fb_pairs   = _collect_pairs(fb_ann_dir)
    if not fb_pairs:
        sys.exit(
            f"No feedback training data at {fb_ann_dir}\n"
            f"Run prepare_parcels.py --sheet {args.sheet} first."
        )

    # ── Replay pool (pre-training annotations from 03_finetune, all sheets) ───
    replay_base = ROOT / paths["annotations"] / "parcel"
    all_replay  = []
    if replay_base.exists():
        for sheet_dir in sorted(replay_base.iterdir()):
            if sheet_dir.is_dir():
                all_replay.extend(_collect_pairs(sheet_dir))

    # ── Build train / val splits ───────────────────────────────────────────────
    rng = np.random.default_rng(seed)

    if all_replay:
        # Shuffle replay pool and split 85 / 15 for train / val
        replay_perm    = rng.permutation(len(all_replay)).tolist()
        n_replay_val   = max(1, int(len(all_replay) * val_split))
        replay_val_idx = set(replay_perm[:n_replay_val])
        replay_train   = [all_replay[i] for i in replay_perm if i not in replay_val_idx]
        replay_val     = [all_replay[i] for i in replay_val_idx]

        # Sample replay_train up to replay_ratio * total_train pairs, cap at 3×
        if replay_ratio > 0 and replay_ratio < 1:
            n_replay = int(len(fb_pairs) * replay_ratio / max(1 - replay_ratio, 1e-6))
            n_replay = min(n_replay, len(replay_train), len(fb_pairs) * 3)
        else:
            n_replay = len(replay_train)

        sampled_replay = [replay_train[i]
                          for i in rng.choice(len(replay_train), n_replay,
                                              replace=n_replay > len(replay_train))]
        train_pairs = fb_pairs + sampled_replay
        val_pairs   = replay_val
        print(f"Feedback  : {len(fb_pairs)} pairs (sheet '{args.sheet}')")
        print(f"Replay    : {n_replay} pairs sampled from {len(all_replay)} "
              f"pre-training annotations")
        print(f"Train mix : {len(train_pairs)} total  "
              f"({len(fb_pairs)} feedback + {n_replay} replay)")
        print(f"Val (cross-sheet): {len(val_pairs)} replay pairs from other sheets")
    else:
        # No pre-training annotations yet — fall back to feedback-only split
        print("  WARNING: no pre-training annotation data found under "
              f"{replay_base}\n"
              "  Falling back to feedback-only train/val split (no cross-sheet "
              "validation).\n"
              "  Run 03_finetune/parcels/train.py on at least one annotated "
              "sheet to enable replay.")
        perm   = rng.permutation(len(fb_pairs)).tolist()
        n_val  = max(1, int(len(fb_pairs) * val_split))
        val_set     = {fb_pairs[i][0].stem for i in perm[:n_val]}
        train_pairs = [p for p in fb_pairs if p[0].stem not in val_set]
        val_pairs   = [p for p in fb_pairs if p[0].stem in val_set]
        print(f"Feedback  : {len(fb_pairs)} pairs → "
              f"{len(train_pairs)} train / {len(val_pairs)} val")

    # ── Dirs ──────────────────────────────────────────────────────────────────
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
        print(f"Weights   : {Path(weights_path).name}  (DoRA resumed)")
    else:
        print(f"Weights   : {Path(weights_path).name}  (SAM base — DoRA fresh)")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    # Shuffle combined train_pairs so feedback and replay are interleaved
    rng.shuffle(train_pairs)

    train_ds = ParcelDataset(train_pairs, img_size, n_bg=n_bg, do_aug=True,
                             min_zoom_area_px=min_zoom_area_px)
    val_ds   = ParcelDataset(val_pairs,   img_size, n_bg=n_bg, do_aug=False)

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
    run_name  = f"sam_parcels_fb_{run_name}"
    best_path = finetuned_dir / f"{run_name}_best.pth"
    log_path  = logs_dir      / f"{run_name}_metrics.csv"

    print(f"\nRun       : {run_name}")
    print(f"Best →      {best_path.relative_to(ROOT)}")
    print(f"Log  →      {log_path.relative_to(ROOT)}\n")

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_iou_cross_sheet"])

    # ── Training loop ─────────────────────────────────────────────────────────
    best_iou   = -1.0
    no_improve = 0
    iter_num   = 0
    low_sz     = img_size // 4

    model.train()
    for epoch in tqdm(range(epochs), desc="Epochs", ncols=70):
        epoch_loss = 0.0

        for batch in train_loader:
            imgs   = batch["image"].to(device)
            coords = batch["coords"].to(device)
            labels = batch["labels"].to(device)
            gt_low = batch["mask_low"].to(device)

            imgs_pre = model.sam.preprocess(imgs * 255.0)
            img_emb  = model.sam.image_encoder(imgs_pre)
            if isinstance(img_emb, tuple):
                img_emb = img_emb[0]

            total_loss = torch.tensor(0.0, device=device)
            for b in range(imgs.shape[0]):
                with torch.no_grad():
                    sp, dp = model.sam.prompt_encoder(
                        points=(coords[b:b+1], labels[b:b+1]),
                        boxes=None, masks=None,
                    )
                lr_masks, iou_preds = model.sam.mask_decoder(
                    image_embeddings=img_emb[b:b+1],
                    image_pe=model.sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sp,
                    dense_prompt_embeddings=dp,
                    multimask_output=True,
                )
                lr_up  = F.interpolate(lr_masks, size=(low_sz, low_sz),
                                        mode="bilinear", align_corners=False)
                gt_exp = gt_low[b:b+1].unsqueeze(1)

                with torch.no_grad():
                    pbin = (torch.sigmoid(lr_up) > 0.5).float()
                    ious = []
                    for mi in range(lr_up.shape[1]):
                        inter = (pbin[:, mi] * gt_exp[:, 0]).sum()
                        union = pbin[:, mi].sum() + gt_exp[:, 0].sum() - inter
                        ious.append(((inter + 1e-6) / (union + 1e-6)).item())
                    best = int(np.argmax(ious))

                total_loss = total_loss + combined_loss(
                    lr_up[:, best:best+1], gt_exp, dice_w)

            total_loss = total_loss / imgs.shape[0]
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            for pg in optimizer.param_groups:
                pg["lr"] = _cosine_lr(iter_num)
            iter_num   += 1
            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        val_iou  = validate(model, val_loader, img_size, device)
        label    = "cross-sheet IoU" if all_replay else "val IoU"

        print(f"  Epoch {epoch+1:3d}/{epochs}  "
              f"loss={avg_loss:.4f}  {label}={val_iou:.4f}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1,
                                    round(avg_loss, 6),
                                    round(val_iou,  4)])

        if val_iou > best_iou:
            best_iou   = val_iou
            no_improve = 0
            model.save_dora_parameters(str(best_path))
            print(f"    ✓ new best {label}={val_iou:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping — best {label}={best_iou:.4f}")
                break

    print(f"\nDone. Best {label}={best_iou:.4f}")
    print(f"  Weights: {best_path.relative_to(ROOT)}")
    print(f"  Log:     {log_path.relative_to(ROOT)}")
    print(f"\nNext:  run parcel_predict.py — it will auto-load the new weights.")


if __name__ == "__main__":
    main()
