"""
Fine-tune MapSAM DoRA weights on pipeline-annotated patches.

Designed for small datasets (5–15 patches) on a laptop GPU (~4 GB).
Adapts the DoRA Q/V projections only — the SAM backbone stays frozen,
keeping memory use and training time low.

The --feature argument accepts any label name used during annotation in labelme.
No feature list needs to be defined in config.yaml — annotation data is read
directly from data/annotations/<feature>/<sheet>/.

Usage
-----
    python train.py --sheet Timberscombe --feature water --name finetune_v1
    python train.py --sheet Timberscombe --feature building --weights models/finetuned/mapsam_building_v1_best.pth

Weight search order (when --weights is not given):
    1. models/finetuned/mapsam_<feature>*_best.pth  (most recent fine-tuned)
    2. models/base/MapSAM/<feature>/                (feature-specific base weights)
    3. models/base/MapSAM/origional_weights/        (generic SAM DoRA fallback)

Data (from step 02_annotate/export_masks.py)
    data/annotations/<feature>/<sheet>/images/*.png  — RGB patch copies
    data/annotations/<feature>/<sheet>/masks/*.png   — binary masks (0/255)

Outputs
    models/finetuned/mapsam_<feature>_<name>_best.pth  — best checkpoint by val IoU
    models/logs/mapsam_<feature>_<name>_metrics.csv    — per-epoch loss + IoU
"""

from __future__ import annotations

import argparse
import csv
import math
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
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT       = Path(__file__).resolve().parents[3]
MAPSAM_DIR = ROOT / "models" / "MapSAM"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MAPSAM_DIR))

from sam_dora_image_encoder import DoRA_Sam      # noqa: E402
from segment_anything import sam_model_registry  # noqa: E402


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class _Augment:
    """
    Matches the original MapSAM training augmentation:
    rot90 + flip OR small rotation (±20°), with optional resize.
    Also produces the 128×128 low_res_label for the intermediate loss term.
    """
    def __init__(self, img_size: int, low_res: int):
        self.img_size = img_size
        self.low_res  = low_res

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        # image: (3, H, W) float32 in [0,1];  mask: (H, W) float32 binary
        if random.random() > 0.5:
            k     = np.random.randint(0, 4)
            image = np.rot90(image, k, axes=(1, 2)).copy()
            mask  = np.rot90(mask, k).copy()
            axis  = np.random.randint(1, 3)
            image = np.flip(image, axis=axis).copy()
            mask  = np.flip(mask, axis=axis - 1).copy()
        elif random.random() > 0.5:
            angle = np.random.randint(-20, 20)
            image = ndimage.rotate(image, angle, order=0, reshape=False)
            mask  = ndimage.rotate(mask,  angle, order=0, reshape=False)

        _, h, w = image.shape
        if h != self.img_size or w != self.img_size:
            image = zoom(image, (1, self.img_size / h, self.img_size / w), order=3)
            mask  = zoom(mask,  (self.img_size / h, self.img_size / w),    order=0)

        lh, lw    = mask.shape
        low_res_m = zoom(mask, (self.low_res / lh, self.low_res / lw), order=0)

        return (
            torch.from_numpy(image.astype(np.float32)),
            torch.from_numpy(mask.astype(np.float32)).long(),
            torch.from_numpy(low_res_m.astype(np.float32)).long(),
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PipelineDataset(Dataset):
    """
    Loads (image, mask) pairs from the pipeline annotation export structure:
        images_dir/<patch>.png  — grayscale or RGB patch
        masks_dir/<patch>.png   — binary mask (0 or 255)

    Accepts a list of (img_path, mask_path) Path tuples so train/val
    splitting by patch name is handled externally.
    """
    def __init__(self, pairs: list, img_size: int, low_res: int):
        self.pairs   = pairs
        self.augment = _Augment(img_size, low_res)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        # SAM expects 3-channel RGB — convert grayscale patches if necessary
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        img = np.transpose(img, (2, 0, 1)) / 255.0  # (3, H, W) in [0, 1]

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = (mask > 127).astype(np.float32)

        image_t, label_t, low_res_t = self.augment(img, mask)
        return {
            'image':         image_t,
            'label':         label_t,
            'low_res_label': low_res_t,
            'case_name':     img_path.stem,
        }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _dice_loss(logits, target, smooth=1e-6):
    probs  = torch.sigmoid(logits)
    flat_p = probs.view(logits.size(0), -1)
    flat_t = target.float().view(target.size(0), -1)
    inter  = (flat_p * flat_t).sum(1)
    return (1. - (2. * inter + smooth) / (flat_p.sum(1) + flat_t.sum(1) + smooth)).mean()


def _focal_loss(inputs, targets, alpha=0.25, gamma=2):
    inputs  = inputs.flatten(1)
    targets = targets.float().flatten(1)
    ce      = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t     = inputs.sigmoid() * targets + (1 - inputs.sigmoid()) * (1 - targets)
    loss    = (alpha * targets + (1 - alpha) * (1 - targets)) * ce * (1 - p_t) ** gamma
    return loss.mean(1).mean()


def _combined_loss(low_res_logits, coarse_mask, target_low_res, dice_w):
    """focal + dice on both the low-res logits and the coarse mask."""
    t = target_low_res.unsqueeze(1)
    return (
        (1 - dice_w) * _focal_loss(low_res_logits, t) + dice_w * _dice_loss(low_res_logits, t) +
        (1 - dice_w) * _focal_loss(coarse_mask,    t) + dice_w * _dice_loss(coarse_mask,    t)
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _validate(model, loader, multimask_output, img_size, threshold=0.5):
    model.eval()
    iou_sum, count = 0.0, 0
    for batch in loader:
        imgs      = batch['image'].cuda()
        low_label = batch['low_res_label'].cuda().float()
        outputs, _ = model(imgs, multimask_output, img_size)
        preds = (torch.sigmoid(outputs['low_res_logits'].squeeze(1)) > threshold)
        for b in range(imgs.shape[0]):
            pred  = preds[b].float()
            label = low_label[b]
            inter = (pred * label).sum()
            union = pred.sum() + label.sum() - inter
            iou_sum += ((inter + 1e-6) / (union + 1e-6)).item()
            count   += 1
    model.train()
    return iou_sum / count if count else 0.0


# ---------------------------------------------------------------------------
# Weights resolution
# ---------------------------------------------------------------------------

def _resolve_weights(args_weights, feature: str, finetuned_dir: Path,
                     mapsam_base_dir: Path) -> str:
    """
    Search order:
      1. --weights CLI argument (explicit override)
      2. Most recent mapsam_<feature>*_best.pth in models/finetuned/
      3. Most recent *.pth in models/base/MapSAM/<feature>/   (feature-specific base)
      4. Most recent *.pth in models/base/MapSAM/origional_weights/  (generic fallback)
    """
    if args_weights:
        p = Path(args_weights)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            return str(p)
        sys.exit(f"Weights not found: {p}")

    # 1. Fine-tuned weights for this specific feature
    candidates = sorted(
        finetuned_dir.rglob(f"mapsam_{feature}*_best.pth"),
        key=lambda p: p.stat().st_mtime,
    )
    if candidates:
        return str(candidates[-1])

    # 2. Feature-specific base weights
    feature_base = mapsam_base_dir / feature
    if feature_base.exists():
        candidates = sorted(feature_base.rglob("*.pth"), key=lambda p: p.stat().st_mtime)
        if candidates:
            return str(candidates[-1])

    # 3. Generic fallback weights (pre-trained MapSAM DoRA, not feature-specific)
    fallback_dir = mapsam_base_dir / "origional_weights"
    if fallback_dir.exists():
        candidates = sorted(fallback_dir.rglob("*.pth"), key=lambda p: p.stat().st_mtime)
        if candidates:
            return str(candidates[-1])

    sys.exit(
        f"No DoRA weights found for feature '{feature}'.\n"
        f"  Searched (finetuned)  : {finetuned_dir}\n"
        f"  Searched (feature)    : {feature_base}\n"
        f"  Searched (fallback)   : {fallback_dir}\n"
        "Pass --weights <path> to specify explicitly."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune MapSAM DoRA weights")
    parser.add_argument("--sheet",   required=True,
                        help="Map sheet name (subdirectory under annotations/)")
    parser.add_argument("--feature", required=True,
                        help="Feature class — any label used in labelme annotations "
                             "(e.g. water, building, vegetation)")
    parser.add_argument("--name",    default=None,
                        help="Run name — prefix for checkpoint and log files")
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--weights", default=None,
                        help="DoRA .pth file to fine-tune from (auto-selects if omitted)")
    args = parser.parse_args()

    cfg    = yaml.safe_load((ROOT / args.config).read_text())
    mcfg   = cfg["mapsam"]
    ft_cfg = mcfg["finetune"]
    paths  = cfg["paths"]

    # Annotation data lives directly under annotations/<feature>/<sheet>/
    # — no per-feature config entry needed
    images_dir = ROOT / paths["annotations"] / args.feature / args.sheet / "images"
    masks_dir  = ROOT / paths["annotations"] / args.feature / args.sheet / "masks"

    # Auto-export masks from labelme JSON if not done yet
    if not images_dir.exists() or not any(images_dir.glob("*.png")):
        print(f"Annotations not found at {images_dir}\nRunning export_masks.py first...\n")
        result = subprocess.run(
            [sys.executable,
             str(ROOT / "steps" / "02_annotate" / "export_masks.py"),
             "--sheet", args.sheet],
            check=False,
        )
        if result.returncode != 0:
            sys.exit("export_masks.py failed — check annotations and try again.")
        print()

    finetuned_dir   = ROOT / paths["models_finetuned"]
    logs_dir        = ROOT / paths["logs"]
    mapsam_base_dir = ROOT / paths["models_base"] / "MapSAM"
    finetuned_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    dora_weights = _resolve_weights(
        args.weights, args.feature, finetuned_dir, mapsam_base_dir
    )
    print(f"Base weights : {Path(dora_weights).name}")

    run_name = args.name or datetime.now().strftime(f"mapsam_{args.feature}_%Y%m%d_%H%M")
    if not run_name.startswith("mapsam"):
        run_name = f"mapsam_{args.feature}_{run_name}"
    best_path = finetuned_dir / f"{run_name}_best.pth"
    log_path  = logs_dir      / f"{run_name}_metrics.csv"

    # Seeds
    seed = ft_cfg.get("seed", 1234)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)

    # ---- Model ----------------------------------------------------------------
    sam_ckpt = (ROOT / paths["models_base"] / "MapSAM"
                / "origional_weights" / "sam_vit_b_01ec64.pth")
    if not sam_ckpt.exists():
        sys.exit(f"SAM base checkpoint not found: {sam_ckpt}")

    sam, img_embed_size = sam_model_registry[mcfg["vit_name"]](
        image_size  = mcfg["img_size"],
        num_classes = mcfg["num_classes"],
        checkpoint  = str(sam_ckpt),
        pixel_mean  = [0, 0, 0],
        pixel_std   = [1, 1, 1],
    )
    net = DoRA_Sam(sam, mcfg["rank"]).cuda()
    net.load_dora_parameters(dora_weights)

    multimask_output = mcfg["num_classes"] > 1
    low_res          = img_embed_size * 4   # 128 for ViT-B

    # ---- Data -----------------------------------------------------------------
    all_pairs = [
        (images_dir / p.name, masks_dir / p.name)
        for p in sorted(images_dir.glob("*.png"))
        if (masks_dir / p.name).exists()
    ]
    if not all_pairs:
        sys.exit(
            f"No paired images + masks found.\n"
            f"  Images: {images_dir}\n"
            f"  Masks:  {masks_dir}"
        )

    names    = [p[0].stem for p in all_pairs]
    rng      = np.random.default_rng(42)
    shuffled = rng.permutation(len(names))
    n_val    = max(1, int(len(names) * ft_cfg["val_split"]))
    val_set  = {names[i] for i in shuffled[:n_val]}

    train_pairs = [p for p in all_pairs if p[0].stem not in val_set]
    val_pairs   = [p for p in all_pairs if p[0].stem in val_set]
    print(f"Patches      : {len(all_pairs)} total → {len(train_pairs)} train / {len(val_pairs)} val")

    train_ds = PipelineDataset(train_pairs, mcfg["img_size"], low_res)
    val_ds   = PipelineDataset(val_pairs,   mcfg["img_size"], low_res)

    train_loader = DataLoader(train_ds, batch_size=ft_cfg["batch_size"],
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=ft_cfg["batch_size"],
                              shuffle=False, num_workers=2, pin_memory=True)

    # ---- Optimiser + schedule ------------------------------------------------
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=ft_cfg["learning_rate"], betas=(0.9, 0.999), weight_decay=0.01,
    )
    max_iters = ft_cfg["epochs"] * len(train_loader)
    dice_w    = mcfg.get("dice_param", 0.8)

    def cosine_lr(i: int) -> float:
        return ft_cfg["learning_rate"] * 0.5 * (1.0 + math.cos(math.pi * i / max(max_iters, 1)))

    # ---- Training loop -------------------------------------------------------
    print(f"\nRun          : {run_name}")
    print(f"Best weights → {best_path.relative_to(ROOT)}")
    print(f"Metrics log  → {log_path.relative_to(ROOT)}\n")

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_iou"])

    best_iou   = -1.0
    no_improve = 0
    patience   = ft_cfg.get("early_stopping_patience", 10)
    iter_num   = 0

    net.train()
    for epoch in tqdm(range(ft_cfg["epochs"]), desc="Epochs", ncols=70):
        epoch_loss = 0.0

        for batch in train_loader:
            imgs      = batch['image'].cuda()
            low_label = batch['low_res_label'].cuda()

            outputs, coarse_mask = net(imgs, multimask_output, mcfg["img_size"])
            loss = _combined_loss(outputs['low_res_logits'], coarse_mask, low_label, dice_w)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = cosine_lr(iter_num)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_

            epoch_loss += loss.item()
            iter_num   += 1

        avg_loss = epoch_loss / len(train_loader)
        val_iou  = _validate(net, val_loader, multimask_output, mcfg["img_size"])

        print(f"\n  Epoch {epoch+1}/{ft_cfg['epochs']}  "
              f"loss={avg_loss:.4f}  val_IoU={val_iou:.4f}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, round(avg_loss, 6), round(val_iou, 4)])

        if val_iou > best_iou:
            best_iou   = val_iou
            no_improve = 0
            net.save_dora_parameters(str(best_path))
            print(f"  ✓ New best val_IoU={val_iou:.4f} — weights saved")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"  Early stopping — best val_IoU={best_iou:.4f}")
                break

    print(f"\nDone. Best val_IoU={best_iou:.4f}")
    print(f"  Weights: {best_path.relative_to(ROOT)}")
    print(f"  Log:     {log_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
