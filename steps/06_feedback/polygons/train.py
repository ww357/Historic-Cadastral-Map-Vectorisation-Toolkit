"""
Feedback fine-tune MapSAM DoRA weights from QGIS-corrected polygon predictions.

Differs from 03_finetune/polygons/train.py in three ways:

  1. Data source: training pairs come from prepare_polygons.py output
     (data/annotations/<feature>/feedback/<sheet>/) — i.e. rasterised QGIS-corrected
     polygons — rather than hand-drawn labelme annotations.

  2. Replay buffer: pre-training annotation patches from all labelled sheets
     (data/annotations/<feature>/<sheet>/ — any sheet dir that is not "feedback/")
     are mixed into every training epoch at replay_ratio.  This prevents the
     SPGen heads and DoRA Q/V weights from drifting away from the multi-sheet
     knowledge built during 03_finetune.

  3. Cross-sheet validation: the validation set is drawn from the replay pool
     (annotations from other sheets), not a split of the current feedback data.
     Early stopping fires when cross-sheet IoU stops improving.

Fallback: if no pre-training annotation data exists the script falls back to
splitting the feedback data 85/15 (feedback-only) with a warning.

MapSAM prompt note:
  This script uses NO external prompts.  SPGen (the Self-Prompt Generator inside
  DoRA_Sam) generates a coarse mask from image embeddings and intermediate ViT
  states; point_selection() inside the forward call extracts foreground/background
  points automatically.  The forward signature is simply:
      outputs, coarse_mask = net(imgs, multimask_output, img_size)

Workflow:
    conda activate polygons
    python steps/06_feedback/polygons/prepare_polygons.py --sheet Timberscombe --feature water
    python steps/06_feedback/polygons/train.py --sheet Timberscombe --feature water

Output:
    models/finetuned/mapsam_<feature>_fb_<name>_best.pth
    models/logs/mapsam_<feature>_fb_<name>_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import math
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


# ── Augmentation ──────────────────────────────────────────────────────────────

class _Augment:
    def __init__(self, img_size: int, low_res: int):
        self.img_size = img_size
        self.low_res  = low_res

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        if random.random() > 0.5:
            k     = np.random.randint(0, 4)
            image = np.rot90(image, k, axes=(1, 2)).copy()
            mask  = np.rot90(mask,  k).copy()
            axis  = np.random.randint(1, 3)
            image = np.flip(image, axis=axis).copy()
            mask  = np.flip(mask,  axis=axis - 1).copy()
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


# ── Dataset ───────────────────────────────────────────────────────────────────

class PolygonDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]],
                 img_size: int, low_res: int, do_aug: bool = True):
        valid = []
        for img_p, mask_p in pairs:
            m = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            if m is not None and (m > 0).any():
                valid.append((img_p, mask_p))
            else:
                print(f"  WARNING: empty mask skipped — {mask_p.name}")
        self.pairs   = valid
        self.augment = _Augment(img_size, low_res) if do_aug else None
        self.img_size = img_size
        self.low_res  = low_res

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
        img = np.transpose(img, (2, 0, 1)) / 255.0   # (3, H, W) in [0, 1]

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        mask = (mask > 127).astype(np.float32)

        if self.augment is not None:
            image_t, label_t, low_res_t = self.augment(img, mask)
        else:
            # No augmentation — just resize and scale
            _, h, w = img.shape
            if h != self.img_size or w != self.img_size:
                img  = zoom(img,  (1, self.img_size / h, self.img_size / w), order=3)
                mask = zoom(mask, (self.img_size / h, self.img_size / w), order=0)
            lh, lw    = mask.shape
            low_res_m = zoom(mask, (self.low_res / lh, self.low_res / lw), order=0)
            image_t   = torch.from_numpy(img.astype(np.float32))
            label_t   = torch.from_numpy(mask.astype(np.float32)).long()
            low_res_t = torch.from_numpy(low_res_m.astype(np.float32)).long()

        return {
            "image":         image_t,
            "label":         label_t,
            "low_res_label": low_res_t,
            "case_name":     img_path.stem,
        }


# ── Loss ──────────────────────────────────────────────────────────────────────

def _dice_loss(logits, target, smooth=1e-6):
    probs  = torch.sigmoid(logits)
    flat_p = probs.view(logits.size(0), -1)
    flat_t = target.float().view(target.size(0), -1)
    inter  = (flat_p * flat_t).sum(1)
    return (1. - (2. * inter + smooth) /
            (flat_p.sum(1) + flat_t.sum(1) + smooth)).mean()


def _focal_loss(inputs, targets, alpha=0.25, gamma=2):
    inputs  = inputs.flatten(1)
    targets = targets.float().flatten(1)
    ce      = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t     = inputs.sigmoid() * targets + (1 - inputs.sigmoid()) * (1 - targets)
    loss    = (alpha * targets + (1 - alpha) * (1 - targets)) * ce * (1 - p_t) ** gamma
    return loss.mean(1).mean()


def _combined_loss(low_res_logits, coarse_mask, target_low_res, dice_w):
    t = target_low_res.unsqueeze(1)
    return (
        (1 - dice_w) * _focal_loss(low_res_logits, t) +
        dice_w       * _dice_loss(low_res_logits,  t) +
        (1 - dice_w) * _focal_loss(coarse_mask,    t) +
        dice_w       * _dice_loss(coarse_mask,      t)
    )


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def _validate(model, loader: DataLoader, multimask_output: bool,
              img_size: int, threshold: float = 0.5) -> float:
    model.eval()
    iou_sum, count = 0.0, 0
    for batch in loader:
        imgs      = batch["image"].cuda()
        low_label = batch["low_res_label"].cuda().float()
        outputs, _ = model(imgs, multimask_output, img_size)
        preds = (torch.sigmoid(outputs["low_res_logits"].squeeze(1)) > threshold)
        for b in range(imgs.shape[0]):
            pred  = preds[b].float()
            label = low_label[b]
            inter = (pred * label).sum()
            union = pred.sum() + label.sum() - inter
            iou_sum += ((inter + 1e-6) / (union + 1e-6)).item()
            count   += 1
    model.train()
    return iou_sum / count if count else 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_pairs(ann_dir: Path) -> list[tuple[Path, Path]]:
    imgs_dir = ann_dir / "images"
    msks_dir = ann_dir / "masks"
    if not imgs_dir.exists():
        return []
    return sorted(
        [(imgs_dir / p.name, msks_dir / p.name)
         for p in imgs_dir.glob("*.png")
         if (msks_dir / p.name).exists()]
    )


def _resolve_weights(args_weights: str | None, feature: str,
                     finetuned_dir: Path, mapsam_base_dir: Path) -> str:
    if args_weights:
        p = Path(args_weights)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            return str(p)
        sys.exit(f"Weights not found: {p}")

    # Prefer most-recent feedback weights for this feature, then any finetune weights
    for pattern in (f"mapsam_{feature}_fb*_best.pth", f"mapsam_{feature}*_best.pth"):
        candidates = sorted(finetuned_dir.rglob(pattern),
                            key=lambda p: p.stat().st_mtime)
        if candidates:
            return str(candidates[-1])

    feature_base = mapsam_base_dir / feature
    if feature_base.exists():
        candidates = sorted(feature_base.rglob("*.pth"), key=lambda p: p.stat().st_mtime)
        if candidates:
            return str(candidates[-1])

    fallback_dir = mapsam_base_dir / "origional_weights"
    if fallback_dir.exists():
        candidates = sorted(fallback_dir.rglob("*.pth"), key=lambda p: p.stat().st_mtime)
        if candidates:
            return str(candidates[-1])

    sys.exit(
        f"No DoRA weights found for feature '{feature}'.\n"
        f"  Searched (finetuned)  : {finetuned_dir}\n"
        f"  Searched (feature)    : {feature_base}\n"
        f"  Searched (fallback)   : {mapsam_base_dir / 'origional_weights'}\n"
        "Pass --weights <path> to specify explicitly."
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feedback fine-tune MapSAM DoRA from QGIS-corrected polygon predictions."
    )
    parser.add_argument("--sheet",   required=True,
                        help="Map sheet name — feedback data must exist in "
                             "data/annotations/<feature>/feedback/<sheet>/")
    parser.add_argument("--feature", required=True,
                        help="Feature class (e.g. water, building, vegetation)")
    parser.add_argument("--name",    default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--config",  default="config.yaml")
    args = parser.parse_args()

    cfg    = yaml.safe_load((ROOT / args.config).read_text())
    paths  = cfg["paths"]
    mcfg   = cfg["mapsam"]
    ft_cfg = mcfg["finetune"]

    img_size  = int(mcfg.get("img_size", 512))
    rank      = int(mcfg.get("rank", 4))
    dice_w    = float(mcfg.get("dice_param", 0.8))
    epochs    = int(ft_cfg.get("epochs", 30))
    bs        = int(ft_cfg.get("batch_size", 2))
    lr        = float(ft_cfg.get("learning_rate", 1e-4))
    val_split = float(ft_cfg.get("val_split", 0.15))
    patience  = int(ft_cfg.get("early_stopping_patience", 10))
    seed      = int(ft_cfg.get("seed", 1234))
    threshold = float(mcfg.get("predict_threshold", 0.5))
    # replay_ratio: read from mapsam.finetune if present, else fall back to feedback section
    replay_ratio = float(
        ft_cfg.get("replay_ratio",
                   cfg.get("feedback", {}).get("replay_ratio", 0.4))
    )

    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device    : {device}")

    # ── Feedback data ─────────────────────────────────────────────────────────
    fb_ann_dir = (ROOT / paths["annotations"]
                  / args.feature / "feedback" / args.sheet)
    fb_pairs   = _collect_pairs(fb_ann_dir)
    if not fb_pairs:
        sys.exit(
            f"No feedback training data at {fb_ann_dir}\n"
            f"Run prepare_polygons.py --sheet {args.sheet} "
            f"--feature {args.feature} first."
        )

    # ── Replay pool (pre-training hand-drawn annotations, all sheets) ─────────
    ann_base   = ROOT / paths["annotations"] / args.feature
    all_replay = []
    if ann_base.exists():
        for sheet_dir in sorted(ann_base.iterdir()):
            # Skip the 'feedback/' subfolder — those are corrected predictions, not GT
            if sheet_dir.is_dir() and sheet_dir.name != "feedback":
                all_replay.extend(_collect_pairs(sheet_dir))

    # ── Build train / val splits ───────────────────────────────────────────────
    rng = np.random.default_rng(seed)

    if all_replay:
        replay_perm    = rng.permutation(len(all_replay)).tolist()
        n_replay_val   = max(1, int(len(all_replay) * val_split))
        replay_val_idx = set(replay_perm[:n_replay_val])
        replay_train   = [all_replay[i] for i in replay_perm if i not in replay_val_idx]
        replay_val     = [all_replay[i] for i in replay_val_idx]

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
        print(f"Feature   : {args.feature}  |  Sheet: {args.sheet}")
        print(f"Feedback  : {len(fb_pairs)} pairs")
        print(f"Replay    : {n_replay} sampled from {len(all_replay)} pre-training annotations")
        print(f"Train mix : {len(train_pairs)} total  "
              f"({len(fb_pairs)} feedback + {n_replay} replay)")
        print(f"Val (cross-sheet): {len(val_pairs)} replay pairs")
    else:
        print("  WARNING: no pre-training annotation data found under "
              f"{ann_base}\n"
              "  Falling back to feedback-only train/val split (no cross-sheet "
              "validation).\n"
              "  Run 03_finetune/polygons/train.py on at least one annotated "
              "sheet first to enable replay.")
        perm   = rng.permutation(len(fb_pairs)).tolist()
        n_val  = max(1, int(len(fb_pairs) * val_split))
        val_set = {fb_pairs[i][0].stem for i in perm[:n_val]}
        train_pairs = [p for p in fb_pairs if p[0].stem not in val_set]
        val_pairs   = [p for p in fb_pairs if p[0].stem in val_set]
        print(f"Feedback  : {len(fb_pairs)} pairs → "
              f"{len(train_pairs)} train / {len(val_pairs)} val")

    # ── Dirs ──────────────────────────────────────────────────────────────────
    finetuned_dir   = ROOT / paths["models_finetuned"]
    logs_dir        = ROOT / paths["logs"]
    mapsam_base_dir = ROOT / paths["models_base"] / "MapSAM"
    finetuned_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    sam_ckpt = mapsam_base_dir / "origional_weights" / "sam_vit_b_01ec64.pth"
    if not sam_ckpt.exists():
        sys.exit(f"SAM base checkpoint not found: {sam_ckpt}")

    sam, img_embed_size = sam_model_registry[mcfg["vit_name"]](
        image_size  = img_size,
        num_classes = mcfg["num_classes"],
        checkpoint  = str(sam_ckpt),
        pixel_mean  = [0, 0, 0],
        pixel_std   = [1, 1, 1],
    )
    net = DoRA_Sam(sam, rank)
    net.to(device)

    dora_weights = _resolve_weights(
        args.weights, args.feature, finetuned_dir, mapsam_base_dir
    )
    net.load_dora_parameters(dora_weights)
    print(f"Weights   : {Path(dora_weights).name}")

    multimask_output = mcfg["num_classes"] > 1
    low_res          = img_embed_size * 4   # 128 for ViT-B

    # ── Run outputs ───────────────────────────────────────────────────────────
    run_name  = args.name or datetime.now().strftime("%Y%m%d_%H%M")
    run_name  = f"mapsam_{args.feature}_fb_{run_name}"
    best_path = finetuned_dir / f"{run_name}_best.pth"
    log_path  = logs_dir      / f"{run_name}_metrics.csv"

    print(f"\nRun       : {run_name}")
    print(f"Best →      {best_path.relative_to(ROOT)}")
    print(f"Log  →      {log_path.relative_to(ROOT)}\n")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    rng.shuffle(train_pairs)

    train_ds = PolygonDataset(train_pairs, img_size, low_res, do_aug=True)
    val_ds   = PolygonDataset(val_pairs,   img_size, low_res, do_aug=False)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=lr, betas=(0.9, 0.999), weight_decay=0.01,
    )
    max_iters = epochs * max(len(train_loader), 1)

    def cosine_lr(i: int) -> float:
        return lr * 0.5 * (1.0 + math.cos(math.pi * i / max(max_iters, 1)))

    # ── Training loop ─────────────────────────────────────────────────────────
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_iou_cross_sheet"])

    best_iou   = -1.0
    no_improve = 0
    iter_num   = 0

    net.train()
    for epoch in tqdm(range(epochs), desc="Epochs", ncols=70):
        epoch_loss = 0.0

        for batch in train_loader:
            imgs      = batch["image"].to(device)
            low_label = batch["low_res_label"].to(device)

            outputs, coarse_mask = net(imgs, multimask_output, img_size)
            loss = _combined_loss(outputs["low_res_logits"], coarse_mask,
                                  low_label, dice_w)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for pg in optimizer.param_groups:
                pg["lr"] = cosine_lr(iter_num)
            iter_num   += 1
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        val_iou  = _validate(net, val_loader, multimask_output, img_size, threshold)
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
            net.save_dora_parameters(str(best_path))
            print(f"    ✓ new best {label}={val_iou:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping — best {label}={best_iou:.4f}")
                break

    print(f"\nDone. Best {label}={best_iou:.4f}")
    print(f"  Weights: {best_path.relative_to(ROOT)}")
    print(f"  Log:     {log_path.relative_to(ROOT)}")
    print(f"\nNext:  run predict.py --feature {args.feature} --sheet {args.sheet} "
          "to re-predict with updated weights.")


if __name__ == "__main__":
    main()
