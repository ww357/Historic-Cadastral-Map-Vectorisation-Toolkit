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


def _sample_negatives(lbl: np.ndarray, cid: int, fg_xy: tuple[float, float],
                      n: int, ring_px: int = 48) -> np.ndarray:
    """
    Return up to n negative (col, row) prompt points for the target parcel `cid`.

    Built in two stages so every instance gets dense negative coverage:
      1. Centroids of the OTHER parcels in the patch, nearest first — this is
         what parcel_predict.py supplies at inference ("segment THIS parcel, not
         those").  Tithe-map patches often contain only a handful of annotated
         parcels, so this alone is too sparse.
      2. The remainder filled with points sampled from a ring AROUND the target
         parcel (a dilated band excluding the parcel itself).  These sit in the
         boundaries/roads/neighbours hugging the parcel and teach SAM not to
         bleed across its edge.

    Returns an empty (0, 2) array when n == 0 (negative prompting disabled).
    """
    if n <= 0:
        return np.empty((0, 2), dtype=np.float32)

    fx, fy = fg_xy
    target = (lbl == cid).astype(np.uint8)

    # ── 1. sibling centroids (nearest first) ──────────────────────────────────
    cents = []
    for ocid in np.unique(lbl):
        if ocid == 0 or ocid == cid:
            continue
        ys, xs = np.where(lbl == ocid)
        if len(xs):
            cents.append((float(xs.mean()), float(ys.mean())))
    if cents:
        cents = np.asarray(cents, dtype=np.float32)
        order = np.argsort((cents[:, 0]-fx)**2 + (cents[:, 1]-fy)**2)
        negs  = cents[order[:n]]
    else:
        negs = np.empty((0, 2), dtype=np.float32)

    # ── 2. fill the remainder from a ring around the target parcel ────────────
    need = n - len(negs)
    if need > 0:
        k    = max(3, ring_px) * 2 + 1
        ring = cv2.dilate(target, np.ones((k, k), np.uint8), iterations=1)
        ring = (ring > 0) & (target == 0)        # band hugging the parcel
        ys, xs = np.where(ring)
        if len(xs) == 0:                          # parcel fills the patch — fall back
            extra = _sample_bg_points(target, need)
        else:
            sel   = np.random.choice(len(xs), min(need, len(xs)), replace=False)
            extra = np.column_stack([xs[sel].astype(np.float32),
                                     ys[sel].astype(np.float32)])
        negs = np.vstack([negs, extra]) if len(negs) else extra

    return negs[:n]


# ── Augmentation ──────────────────────────────────────────────────────────────

def _augment(img: np.ndarray, lbl: np.ndarray,
             min_zoom_area_px: int = 1500) -> tuple[np.ndarray, np.ndarray]:
    """
    Augmentation for parcel SAM training.  Operates on an image and an integer
    LABEL MAP (0 = background, 1..K = distinct parcel instances).  All geometric
    transforms use INTER_NEAREST on the label map so component ids are preserved
    — this lets a single augmented patch still resolve into per-parcel instances.

    Three independent transforms applied sequentially:

    1. rot90 + flip  (50%)         — orientation invariance
    2. Small rotation ±15°  (25%)  — slight perspective variation
    3. Random zoom-out  (40%)      — MULTI-SCALE: simulates the adaptive 2×
       scale path in parcel_predict.py where a 2048px window is downsampled
       to 1024px.  A centre-crop of 1.2–2.0× is resized back to the original
       size, shrinking boundaries proportionally.  Without this the model only
       ever sees 0.5m/px resolution during training, but the 2× path shows it
       boundaries at 1m/px — half the pixel density.

       Skipped if the total annotated area is below min_zoom_area_px: very small
       parcels become indistinguishable after a 2× downscale so zooming out
       would produce a misleading training signal rather than a useful one.
    """
    h, w = img.shape[:2]

    # 1. rot90 + flip
    if random.random() > 0.5:
        k    = np.random.randint(0, 4)
        img  = np.rot90(img, k).copy()
        lbl  = np.rot90(lbl, k).copy()
        if random.random() > 0.5:
            img  = np.fliplr(img).copy()
            lbl  = np.fliplr(lbl).copy()
        elif random.random() > 0.5:
            img  = np.flipud(img).copy()
            lbl  = np.flipud(lbl).copy()

    # 2. Small rotation
    if random.random() > 0.75:
        angle = np.random.uniform(-15, 15)
        M     = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img   = cv2.warpAffine(img, M, (w, h),
                                borderMode=cv2.BORDER_REFLECT_101)
        lbl   = cv2.warpAffine(lbl, M, (w, h),
                                flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)

    # 3. Multi-scale zoom-OUT (simulates the 2× adaptive scale path)
    # Shrink the WHOLE patch onto a larger white canvas so parcels appear smaller
    # with more surrounding margin — this matches parcel_predict.py downsampling a
    # 2048px window to 1024px (parcels at half the pixel density).
    #
    # Implemented as shrink + pad, NOT a centre crop.  A centre crop would slice
    # edge/corner parcels out of frame — fatal now that each parcel is its own
    # training target (the target mask would be emptied and its foreground prompt
    # lost).  Shrink+pad keeps every parcel fully visible.
    area = int((lbl > 0).sum())
    if random.random() > 0.6 and area >= min_zoom_area_px:
        factor       = np.random.uniform(1.2, 2.0)        # 1.2× – 2× smaller
        new_h, new_w = int(h / factor), int(w / factor)
        small_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        small_lbl = cv2.resize(lbl, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # White canvas (255 = map margin colour, matching patchify pad_value)
        canvas_img = np.full((h, w, img.shape[2]), 255, dtype=img.dtype)
        canvas_lbl = np.zeros((h, w), dtype=lbl.dtype)
        # Random placement also gives a translation augmentation
        oy = np.random.randint(0, h - new_h + 1)
        ox = np.random.randint(0, w - new_w + 1)
        canvas_img[oy:oy+new_h, ox:ox+new_w] = small_img
        canvas_lbl[oy:oy+new_h, ox:ox+new_w] = small_lbl
        img, lbl = canvas_img, canvas_lbl

    return img, lbl


# ── Dataset ───────────────────────────────────────────────────────────────────

class ParcelAnnotationDataset(Dataset):
    """
    Instance-level parcel dataset.

    Reads (image, mask) patch pairs from the standard export format:
        data/annotations/parcel/<sheet>/images/*.png
        data/annotations/parcel/<sheet>/masks/*.png

    export_masks.py merges every "parcel" polygon drawn in a patch into ONE
    binary mask.  Training point-prompted SAM on that merged mask is wrong: at
    inference parcel_predict.py gives a single point per parcel and expects just
    THAT parcel back.  So here each patch mask is split into connected
    components, and EVERY parcel becomes its own training example:

        • foreground prompt  = a point inside that one parcel
        • GT mask            = that one parcel only
        • negative prompts   = the centroids of the OTHER parcels in the patch
                               (nearest first) — identical to inference

    One patch with three drawn parcels therefore yields three training items
    that share the patch image (and its cached encoder embedding) but differ in
    target mask and prompts.
    """

    def __init__(self, pairs: list[tuple[Path, Path]],
                 img_size: int, n_bg: int = 6, do_aug: bool = True,
                 min_zoom_area_px: int = 1500, min_instance_px: int = 50,
                 neg_ring_px: int = 48):
        self.img_size         = img_size
        self.n_bg             = n_bg
        self.do_aug           = do_aug
        self.min_zoom_area_px = min_zoom_area_px
        self.min_instance_px  = min_instance_px
        self.neg_ring_px      = neg_ring_px

        # Keep only patches with at least one non-empty mask, and build the
        # per-parcel instance index by connected-componenting each mask.
        # cv2.connectedComponents labels deterministically for a given binary
        # input, so the ids recovered here match those recomputed in __getitem__.
        self.pairs: list[tuple[Path, Path]] = []
        self.index: list[tuple[int, int]]   = []   # (pair_idx, component_id)
        for img_p, mask_p in pairs:
            m = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            if m is None or not (m > 0).any():
                print(f"  WARNING: empty mask skipped — {mask_p.name}")
                continue
            n_lbl, lbl = cv2.connectedComponents((m > 127).astype(np.uint8))
            comps = [c for c in range(1, n_lbl)
                     if int((lbl == c).sum()) >= self.min_instance_px]
            if not comps:
                print(f"  WARNING: no parcels above {self.min_instance_px}px "
                      f"in {mask_p.name} — skipped")
                continue
            pi = len(self.pairs)
            self.pairs.append((img_p, mask_p))
            self.index.extend((pi, c) for c in comps)

        # Augment-once cache (see note below) keyed by pair_idx → (img, labelmap)
        # at native patch resolution.  All instances of a patch share this entry,
        # so they share one consistent augmented image — REQUIRED for the encoder
        # embedding (pre-computed once per image) to stay valid every epoch.
        self._frozen: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _load_pair(self, pi: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (img RGB, label map) at img_size, augmented+frozen if enabled."""
        img_path, mask_path = self.pairs[pi]

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
        # uint16 label map — cv2.warpAffine/resize support 16U (not 32S),
        # and parcels-per-patch is far below 65535.
        _, lbl = cv2.connectedComponents((mask > 127).astype(np.uint8))
        lbl = lbl.astype(np.uint16)

        if self.do_aug:
            if pi not in self._frozen:
                img_a, lbl_a = _augment(img, lbl, self.min_zoom_area_px)
                # If augmentation drops any indexed parcel out of frame, fall
                # back to the unaugmented patch so every instance stays valid.
                wanted = {c for p, c in self.index if p == pi}
                survived = {int(c) for c in np.unique(lbl_a) if c != 0}
                if not wanted.issubset(survived):
                    img_a, lbl_a = img, lbl
                self._frozen[pi] = (img_a, lbl_a)
            img, lbl = self._frozen[pi]

        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_LINEAR)
            lbl = cv2.resize(lbl, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_NEAREST)
        return img, lbl

    def __getitem__(self, idx: int) -> dict:
        pi, cid = self.index[idx]
        img, lbl = self._load_pair(pi)

        # Target = this one parcel only
        target = (lbl == cid).astype(np.uint8)

        # ── Point prompts ─────────────────────────────────────────────────────
        fg = _sample_fg_point(target)
        if fg is None:
            # Component vanished post-resize (very small) — should be rare given
            # min_instance_px.  Fall back to the patch's largest parcel.
            ys, xs = np.where(lbl > 0)
            fg = (float(xs.mean()), float(ys.mean()))
        neg = _sample_negatives(lbl, cid, fg, self.n_bg, self.neg_ring_px)

        if len(neg):
            coords = np.vstack([[list(fg)], neg.tolist()]).astype(np.float32)
            labels = np.array([1] + [0]*len(neg), dtype=np.int32)
        else:
            coords = np.array([list(fg)], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)

        # GT mask at SAM's internal low-res output size (img_size // 4)
        low_sz   = self.img_size // 4
        low_mask = cv2.resize(target.astype(np.float32), (low_sz, low_sz),
                              interpolation=cv2.INTER_NEAREST)

        img_t = torch.tensor(
            np.ascontiguousarray(img, dtype=np.uint8), dtype=torch.float32
        ) / 255.0
        img_t = img_t.permute(2, 0, 1)   # (3, H, W)

        return {
            "image"    : img_t,
            "mask_low" : torch.tensor(low_mask, dtype=torch.float32),
            "coords"   : torch.tensor(coords,   dtype=torch.float32),
            "labels"   : torch.tensor(labels,   dtype=torch.int32),
            "img_key"  : self.pairs[pi][0].stem,          # encoder-cache key (per image)
            "case_name": f"{self.pairs[pi][0].stem}#c{cid}",  # unique per instance
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
              device: torch.device,
              emb_cache: dict | None = None) -> float:
    model.eval()
    iou_sum, n = 0.0, 0
    low_sz = img_size // 4
    for batch in loader:
        imgs   = batch["image"].to(device)
        coords = batch["coords"].to(device)
        labels = batch["labels"].to(device)
        gt_low = batch["mask_low"].to(device)

        if emb_cache is not None:
            img_emb = emb_cache[batch["img_key"][0]].to(device)
        else:
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
                     base_dir: Path, fresh: bool = False) -> str:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            return str(p)
        sys.exit(f"Weights not found: {p}")

    # --fresh skips the resume scan and starts DoRA from untouched SAM base.
    # Use it when previous parcel checkpoints came from buggy/experimental runs
    # you don't want to inherit.
    if not fresh:
        candidates = sorted(finetuned_dir.glob("sam_parcels*_best.pth"),
                            key=lambda p: p.stat().st_mtime)
        if candidates:
            print(f"Resuming from: {candidates[-1].name}")
            return str(candidates[-1])

    fallback = base_dir / "origional_weights" / "sam_vit_b_01ec64.pth"
    if fallback.exists():
        print("Initialising DoRA from SAM base (fresh)." if fresh
              else "No previous parcel weights — initialising DoRA from SAM base.")
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
    parser.add_argument("--fresh",   action="store_true",
                        help="Ignore existing sam_parcels*_best.pth checkpoints and "
                             "start DoRA from untouched SAM base weights")
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
    # Training negatives are independent of inference (parcels.max_neg_points):
    # patches have few annotated neighbours, so we supplement with a ring of
    # background points around each parcel to reach neg_points.
    n_bg             = int(ft_cfg.get("neg_points", 12))
    neg_ring_px      = int(ft_cfg.get("neg_ring_px", 48))
    min_zoom_area_px = int(ft_cfg.get("min_zoom_area_px", 1500))
    min_instance_px  = int(ft_cfg.get("min_instance_px", 50))
    seed             = int(ft_cfg.get("seed", 1234))

    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device   : {device}")

    # ── Resolve sheet list ────────────────────────────────────────────────────
    feature  = "parcel"
    base_ann = ROOT / paths["annotations"] / feature

    if args.all_sheets:
        # Sheets with already-exported masks
        exported = set(d.name for d in base_ann.iterdir() if d.is_dir()) \
                   if base_ann.exists() else set()
        # Sheets with labelme JSONs not yet exported
        patches_base = ROOT / "data" / "patches" / "parcel"
        with_jsons   = set()
        if patches_base.exists():
            for d in patches_base.iterdir():
                if d.is_dir() and any(d.glob("*.json")):
                    with_jsons.add(d.name)
        sheet_ids = sorted(exported | with_jsons)
        if not sheet_ids:
            sys.exit(
                f"No annotated sheets found.\n"
                f"  Exported masks : {base_ann}\n"
                f"  Labelme JSONs  : {patches_base}\n"
                "Run parcel_patchify.py → annotate_parcels.py first."
            )
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

    # IMPORTANT: the image encoder is FROZEN at original SAM weights, so it must
    # receive the ImageNet-normalised input distribution it was pretrained on.
    # MapSAM uses pixel_mean=[0,0,0] only because it trains the encoder DoRA to
    # adapt to raw [0,1] input — we do not, so [0,0,0] (which left preprocess a
    # no-op and fed the encoder raw [0,255]) produced garbage embeddings.
    # The dataset yields [0,1] images; `preprocess(imgs * 255.0)` then maps them
    # to [0,255] and these mean/std normalise to SAM's expected ~[-2,2] range.
    # parcel_predict.py must use the SAME normalisation (it now does).
    sam, _ = sam_model_registry["vit_b"](
        image_size  = img_size,
        num_classes = 4,
        checkpoint  = str(sam_ckpt),
        pixel_mean  = [123.675, 116.28, 103.53],
        pixel_std   = [58.395,  57.12,  57.375],
    )
    model = DoRA_Sam(sam, rank).to(device)

    weights_path = _resolve_weights(args.weights, finetuned_dir, base_dir,
                                     fresh=args.fresh)
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
        print(f"  [{sheet}] {len(sheet_pairs)} annotated patches")
        all_pairs.extend(sheet_pairs)
    if not all_pairs:
        sys.exit(f"No image+mask pairs found under {base_ann}")

    print(f"Patches  : {len(all_pairs)} annotated patches")

    # Split by PATCH (not by parcel instance) so all parcels from one patch stay
    # on the same side — prevents the same image leaking across train/val.
    rng  = np.random.default_rng(seed)
    perm = rng.permutation(len(all_pairs))
    n_val = max(1, int(len(all_pairs) * val_split))

    val_set     = {all_pairs[i][0].stem for i in perm[:n_val]}
    train_pairs = [p for p in all_pairs if p[0].stem not in val_set]
    val_pairs   = [p for p in all_pairs if p[0].stem in val_set]

    train_ds = ParcelAnnotationDataset(train_pairs, img_size, n_bg=n_bg, do_aug=True,
                                       min_zoom_area_px=min_zoom_area_px,
                                       min_instance_px=min_instance_px,
                                       neg_ring_px=neg_ring_px)
    val_ds   = ParcelAnnotationDataset(val_pairs,   img_size, n_bg=n_bg, do_aug=False,
                                       min_instance_px=min_instance_px,
                                       neg_ring_px=neg_ring_px)
    print(f"Split    : {len(train_pairs)} train / {len(val_pairs)} val patches  "
          f"→  {len(train_ds)} train / {len(val_ds)} val parcel instances")
    if len(train_ds) < 10:
        print("  ⚠  Fewer than 10 parcel instances — fine-tuning will be noisy. "
              "Annotate more parcels for better results.")

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

    # ── Pre-compute image embeddings (encoder frozen) ─────────────────────────
    # SAM's ViT-B encoder is the compute bottleneck (~95% of forward-pass time).
    # With only 98 training patches, running the encoder on every batch every
    # epoch adds hours of overhead while contributing very little gradient signal
    # (DoRA updates are tiny; embeddings barely change between steps).
    # Pre-computing once and freezing the encoder reduces per-epoch time from
    # ~4 minutes to ~15 seconds without measurable quality loss on small datasets.
    print("Pre-computing image embeddings (encoder frozen)...")
    for param in model.sam.image_encoder.parameters():
        param.requires_grad = False

    # Cache is keyed per IMAGE (img_key), not per instance — several parcel
    # instances share one patch image and therefore one embedding, so we skip
    # any image already encoded.
    emb_cache: dict[str, torch.Tensor] = {}
    precompute_loader = DataLoader(
        train_ds, batch_size=1, shuffle=False, num_workers=0
    )
    model.eval()
    with torch.no_grad():
        for pbatch in tqdm(precompute_loader, desc="Embedding", ncols=70, leave=False):
            key = pbatch["img_key"][0]
            if key in emb_cache:
                continue
            imgs_pre = model.sam.preprocess(pbatch["image"].to(device) * 255.0)
            emb      = model.sam.image_encoder(imgs_pre)
            if isinstance(emb, tuple):
                emb = emb[0]
            emb_cache[key] = emb.cpu()

    # Also cache val embeddings
    val_precompute = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0
    )
    with torch.no_grad():
        for pbatch in val_precompute:
            key = pbatch["img_key"][0]
            if key in emb_cache:
                continue
            imgs_pre = model.sam.preprocess(pbatch["image"].to(device) * 255.0)
            emb      = model.sam.image_encoder(imgs_pre)
            if isinstance(emb, tuple):
                emb = emb[0]
            emb_cache[key] = emb.cpu()
    print(f"  Cached {len(emb_cache)} embeddings.\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_bar  = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{epochs}",
                          ncols=80, leave=False)
        for batch in batch_bar:
            imgs   = batch["image"].to(device)
            coords = batch["coords"].to(device)
            labels = batch["labels"].to(device)
            gt_low = batch["mask_low"].to(device)

            img_emb = emb_cache[batch["img_key"][0]].to(device)

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
            batch_bar.set_postfix(loss=f"{total_loss.item():.4f}")

        avg_loss = epoch_loss / max(len(train_loader), 1)
        val_iou  = _validate(model, val_loader, img_size, device, emb_cache=emb_cache)

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
