"""
Feedback fine-tuning for the boundary U-Net.

Reads the dataset manifest and mixes three tile sources per training batch:
  - Original training data  (source=original,    tier=ground_truth, weight=1.0)
  - Hand-drawn annotations  (source=annotation,  tier=ground_truth, weight=1.0)
  - Feedback pseudo-labels  (source=feedback,    tier=pseudo_label, weight=feedback_loss_weight)

Catastrophic forgetting is mitigated by three mechanisms applied together:
  1. Replay buffer: ground_truth tiles from ALL sheets mixed into every batch
     (replay_ratio controls the fraction from the replay pool vs feedback tiles)
  2. Frozen encoder: the first N encoder blocks are frozen so low-level feature
     detectors trained on diverse maps are not overwritten
  3. Cross-sheet early stopping: validation set uses ground_truth tiles from
     sheets OTHER than the current feedback sheet; training halts when the
     model starts to forget them, not just when feedback loss plateaus

Weight search order (starting weights):
  1. --weights CLI argument (explicit path)
  2. Most recently modified feedback_v* in models/finetuned/iterative/
  3. Most recently modified *.weights.h5 in models/base/ (recursive)

Note: step 03 working weights (models/finetuned/working/) are deliberately
excluded from this search — feedback training always starts from the iterative
weights (general model), not from the sheet-specific working weights.

Reads  : data/training/boundary_dataset/manifest.csv
         data/training/boundary_dataset/{train,annotation/}*.png

Writes : models/finetuned/iterative/feedback_vN_best.weights.h5  (auto-incremented)
         models/logs/feedback_vN_metrics.csv

Usage:
    conda activate tf-gpu
    python steps/06_feedback/lines/train.py --sheet SHEET_ID
    python steps/06_feedback/lines/train.py --sheet SHEET_ID --name my_run

Options:
    --sheet      Sheet whose feedback tiles form the pseudo-label training set
    --name       Override the auto-incremented run name (feedback_v1, v2, ...)
    --weights    Explicit starting weights path (overrides auto-selection)
    --no-freeze  Disable encoder freezing (try if model is underfitting)
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.ImprovedLinearUNet.architecture import (
    build_model, LOSS_MAP, dsc, clDice, tp, tn, prec, recall,
)
from models.ImprovedLinearUNet.augmentation import BoundaryAugmentation
from models.ImprovedLinearUNet.metrics import compute_path_metrics, MENDING_C, MENDING_K

import tensorflow as tf
from tensorflow.keras.optimizers import Adam


# ---------------------------------------------------------------------------
# Encoder freezing
# ---------------------------------------------------------------------------

# Layer name prefixes per encoder block (matches build_attn_unet in architecture.py)
_ENCODER_BLOCK_PREFIXES = [
    ("conv1", "drop1"),  # Block 1: 32 filters, full resolution
    ("conv2", "drop2"),  # Block 2: 32 filters, /2
    ("conv3", "drop3"),  # Block 3: 64 filters, /4
    ("conv4", "drop4"),  # Block 4: 64 filters, /8
]


def freeze_encoder_blocks(model, n_blocks: int) -> int:
    """Freeze the first n_blocks encoder stages.  Must recompile after calling.
    Returns the count of layers that were frozen.
    """
    if n_blocks <= 0:
        return 0
    prefixes: set[str] = set()
    for i in range(min(n_blocks, len(_ENCODER_BLOCK_PREFIXES))):
        prefixes.update(_ENCODER_BLOCK_PREFIXES[i])

    n_frozen = 0
    for layer in model.layers:
        if any(layer.name.startswith(p) for p in prefixes):
            layer.trainable = False
            n_frozen += 1
    return n_frozen


# ---------------------------------------------------------------------------
# Tile loading
# ---------------------------------------------------------------------------

def load_tiles(
    manifest_rows: list[dict],
    dataset_dir: Path,
    tile_size: int,
    *,
    split: str | None = None,
    sheet_eq: str | None = None,
    sheet_ne: str | None = None,
    tier: str | None = None,
    source: str | None = None,
) -> list[dict]:
    """Load tiles matching all supplied filter kwargs.
    Returns list of dicts: {img_tile, mask_tile, sheet, tier, source}.
    """
    records = []
    for row in manifest_rows:
        if split    is not None and row["split"]  != split:    continue
        if sheet_eq is not None and row["sheet"]  != sheet_eq: continue
        if sheet_ne is not None and row["sheet"]  == sheet_ne: continue
        if tier     is not None and row["tier"]   != tier:     continue
        if source   is not None and row["source"] != source:   continue

        img_path  = dataset_dir / row["image_path"]
        mask_path = dataset_dir / row["mask_path"]
        if not img_path.exists() or not mask_path.exists():
            continue

        img  = np.array(Image.open(img_path) .convert("L"), dtype=np.float32) / 255.0
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = (mask > 127).astype(np.float32)

        records.append({
            "img_tile":  img,
            "mask_tile": mask,
            "sheet":     row["sheet"],
            "tier":      row["tier"],
            "source":    row["source"],
        })
    return records


# ---------------------------------------------------------------------------
# Mixed training dataset
# ---------------------------------------------------------------------------

def build_training_arrays(
    gt_tiles: list[dict],
    fb_tiles: list[dict],
    replay_ratio: float,
    fb_weight: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build mixed (X, y, w) numpy arrays for training.

    replay_ratio controls what fraction of each batch comes from ground_truth.
    n_gt is scaled so the ratio is approximately met:
      replay_ratio = n_gt / (n_gt + n_fb)
      → n_gt = n_fb * replay_ratio / (1 - replay_ratio)

    GT tiles are sampled with replacement if n_gt > len(gt_tiles) (allowing
    each GT example to appear multiple times in one epoch — same as oversampling).
    Capped at 5× to avoid a single GT-dominated epoch.

    Returns X [N,256,256,1], y [N,256,256,1], w [N] (float32 sample weights).
    """
    rng = np.random.default_rng(seed)
    n_fb = len(fb_tiles)

    if n_fb == 0:
        sys.exit(
            "No feedback tiles found for the current sheet.\n"
            "Run rasterise.py first:\n"
            "  conda activate maptools\n"
            "  python steps/06_feedback/lines/rasterise.py --sheet SHEET_ID"
        )

    if gt_tiles and 0 < replay_ratio < 1:
        n_gt = int(n_fb * replay_ratio / (1.0 - replay_ratio))
        n_gt = min(n_gt, len(gt_tiles) * 5)
        replace = n_gt > len(gt_tiles)
        gt_idx = rng.choice(len(gt_tiles), n_gt, replace=replace)
        sampled_gt = [gt_tiles[i] for i in gt_idx]
    elif replay_ratio >= 1:
        sampled_gt = list(gt_tiles)
    else:
        sampled_gt = []

    all_tiles = sampled_gt + list(fb_tiles)
    X = np.stack([t["img_tile"]  for t in all_tiles])[..., np.newaxis]
    y = np.stack([t["mask_tile"] for t in all_tiles])[..., np.newaxis]
    w = np.array(
        [1.0] * len(sampled_gt) + [fb_weight] * n_fb, dtype=np.float32
    )

    idx = rng.permutation(len(X))
    return X[idx], y[idx], w[idx]


# ---------------------------------------------------------------------------
# Weighted augmented tf.data.Dataset (supports sample weights)
# ---------------------------------------------------------------------------

def make_weighted_augmented_dataset(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    batch_size: int,
    tile_size: int,
) -> tf.data.Dataset:
    """tf.data.Dataset that yields (image_batch, mask_batch, weight_batch).
    Keras uses the third element automatically as per-sample loss weights.
    """
    aug = BoundaryAugmentation(patch_size=tile_size, probability=0.7, occlusion_p=0.6)

    def _augment(img, mask, weight):
        img_aug, mask_aug = aug(img, mask)
        return img_aug, mask_aug, weight

    ds = (
        tf.data.Dataset
        .from_tensor_slices((X, y, w))
        .shuffle(buffer_size=len(X))
        .map(
            lambda x, y, w: (tf.cast(x, tf.float32), tf.cast(y, tf.float32), w),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds


# ---------------------------------------------------------------------------
# Cross-sheet Path-F1 callback
# ---------------------------------------------------------------------------

class CrossSheetCallback(tf.keras.callbacks.Callback):
    """
    After each epoch, evaluate Path-F1 on a held-out set of ground_truth tiles
    from sheets OTHER than the current feedback sheet.  This directly measures
    whether the model is retaining capability on maps it was not fine-tuned on.

    Saves weights only when cross-sheet Path-F1 improves.
    Early stops when it has not improved for `patience` epochs.
    """

    def __init__(
        self,
        val_tiles: list[dict],
        tau: int,
        threshold: float,
        best_weights_path: Path,
        log_path: Path,
        max_val_tiles: int = 200,
        patience: int = 5,
    ):
        super().__init__()
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(val_tiles))[:max_val_tiles]
        self.val_tiles          = [val_tiles[i] for i in idx]
        self.tau                = tau
        self.threshold          = threshold
        self.best_weights_path  = best_weights_path
        self.log_path           = log_path
        self.patience           = patience
        self.best_f1            = -1.0
        self._no_improve        = 0
        self._init_log()

    def _init_log(self):
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss",
                "crosssheet_apl", "crosssheet_fn", "crosssheet_fp",
                "crosssheet_recall", "crosssheet_precision", "crosssheet_path_f1",
                "est_mending_time_min",
            ])

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        per_tile = []
        for rec in self.val_tiles:
            inp  = rec["img_tile"][np.newaxis, ..., np.newaxis]
            pred = self.model.predict(inp, verbose=0)[0, ..., 0]
            m = compute_path_metrics(
                pred_mask=(pred >= self.threshold).astype(np.uint8),
                gt_mask  =(rec["mask_tile"] >= 0.5).astype(np.uint8),
                tau      =self.tau,
            )
            per_tile.append(m)

        path_f1   = float(np.mean([m["path_f1"]        for m in per_tile]))
        path_rec  = float(np.mean([m["path_recall"]    for m in per_tile]))
        path_prec = float(np.mean([m["path_precision"] for m in per_tile]))
        apl_total = sum(m["apl"]       for m in per_tile)
        fn_total  = sum(m["fn_length"] for m in per_tile)
        fp_total  = sum(m["fp_length"] for m in per_tile)
        est_time  = MENDING_C * (apl_total ** MENDING_K) if apl_total > 0 else 0.0

        print(
            f"\n  Cross-sheet Path-F1: {path_f1:.4f}  "
            f"(R={path_rec:.3f}  P={path_prec:.3f})  "
            f"APL: {apl_total:.0f}px  "
            f"Est. mending: {est_time:.1f} min  (relative only, R²=0.317)"
        )

        if path_f1 > self.best_f1:
            self.best_f1     = path_f1
            self._no_improve = 0
            self.model.save_weights(str(self.best_weights_path))
            print(f"  ✓ New best cross-sheet path_f1={path_f1:.4f} — weights saved")
        else:
            self._no_improve += 1
            print(f"  No improvement ({self._no_improve}/{self.patience})")
            if self._no_improve >= self.patience:
                print(
                    f"  Early stopping — cross-sheet Path-F1 not improving.\n"
                    f"  Best={self.best_f1:.4f}  "
                    f"(model is drifting away from generalised performance)"
                )
                self.model.stop_training = True

        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1,
                round(logs.get("loss", 0),   6),
                round(apl_total, 1), round(fn_total, 1), round(fp_total, 1),
                round(path_rec,  4), round(path_prec, 4), round(path_f1,  4),
                round(est_time,  2),
            ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Feedback fine-tune the boundary U-Net")
    parser.add_argument("--sheet",    required=True,
                        help="Sheet ID whose feedback tiles are the pseudo-label training set")
    parser.add_argument("--name",     default=None,
                        help="Run name for checkpoint and log files")
    parser.add_argument("--weights",  default=None,
                        help="Starting weights path (default: auto-select latest finetuned)")
    parser.add_argument("--no-freeze", action="store_true",
                        help="Disable encoder freezing")
    args = parser.parse_args()

    sheet_id = args.sheet

    cfg      = yaml.safe_load((ROOT / "config.yaml").read_text())
    unet_cfg = cfg["unet"]
    ft_cfg   = cfg["finetune"]
    fb_cfg   = cfg.get("feedback", {})
    paths    = cfg["paths"]

    tile_size     = int(unet_cfg["inference_size"])    # 256
    batch_size    = int(unet_cfg["batch_size"])
    threshold     = float(unet_cfg["threshold"])
    loss_type     = unet_cfg["loss_type"]

    replay_ratio  = float(fb_cfg.get("replay_ratio",          0.4))
    fb_weight     = float(fb_cfg.get("feedback_loss_weight",  0.8))
    freeze_n      = int  (fb_cfg.get("freeze_encoder_blocks", 2))   if not args.no_freeze else 0
    fb_lr         = float(fb_cfg.get("learning_rate",         1e-5))
    epochs        = int  (fb_cfg.get("epochs",                20))
    patience      = int  (fb_cfg.get("early_stopping_patience", 5))
    max_val_tiles = int  (fb_cfg.get("max_val_tiles",         200))
    apl_tau       = int  (ft_cfg.get("apl_tau",               3))

    dataset_dir   = ROOT / "data" / "training" / "boundary_dataset"
    iterative_dir = ROOT / paths["models_finetuned"] / "iterative"
    logs_dir      = ROOT / paths["logs"]
    iterative_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Auto-increment feedback version unless --name is supplied
    if args.name:
        run_name = args.name
    else:
        import re
        existing = list(iterative_dir.glob("feedback_v*_best.weights.h5"))
        nums = []
        for p in existing:
            m = re.match(r"feedback_v(\d+)_best\.weights\.h5", p.name)
            if m:
                nums.append(int(m.group(1)))
        next_n   = max(nums) + 1 if nums else 1
        run_name = f"feedback_v{next_n}"

    # ---- Load manifest ------------------------------------------------------
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.exists():
        sys.exit(
            f"manifest.csv not found at {manifest_path}\n"
            "Run rasterise.py first:\n"
            "  conda activate maptools\n"
            f"  python steps/06_feedback/lines/rasterise.py --sheet {sheet_id}"
        )
    manifest = pd.read_csv(manifest_path).to_dict("records")
    print(f"Manifest loaded: {len(manifest)} entries")

    # ---- Load tile sets -----------------------------------------------------
    print("\nLoading tiles...")

    # Ground-truth pool: original + annotation, all sheets (replay buffer)
    gt_tiles = load_tiles(manifest, dataset_dir, tile_size,
                          split="train", tier="ground_truth")

    # Feedback tiles: pseudo-labels for the current sheet only
    fb_tiles = load_tiles(manifest, dataset_dir, tile_size,
                          split="train", source="feedback", sheet_eq=sheet_id)

    # Cross-sheet validation: ground_truth tiles from ALL OTHER sheets
    # Use test-split tiles first (held-out from original training); supplement
    # with train-split tiles from other sheets if test-split is small.
    val_tiles_test = load_tiles(manifest, dataset_dir, tile_size,
                                split="test", tier="ground_truth", sheet_ne=sheet_id)
    if len(val_tiles_test) < 20:
        # Supplement from train split of other sheets (random 20% sample)
        extra = load_tiles(manifest, dataset_dir, tile_size,
                           split="train", tier="ground_truth", sheet_ne=sheet_id)
        rng_val = np.random.default_rng(1)
        n_extra = min(len(extra), max(0, 100 - len(val_tiles_test)))
        idx     = rng_val.choice(len(extra), n_extra, replace=False)
        val_tiles_test += [extra[i] for i in idx]

    val_tiles = val_tiles_test

    # Check
    fb_sheets = {t["sheet"] for t in fb_tiles}
    gt_sheets = {t["sheet"] for t in gt_tiles}
    val_sheets = {t["sheet"] for t in val_tiles}

    print(f"  Ground-truth pool : {len(gt_tiles):>5} tiles  "
          f"from {len(gt_sheets)} sheets")
    print(f"  Feedback tiles    : {len(fb_tiles):>5} tiles  "
          f"for sheet '{sheet_id}'")
    print(f"  Cross-sheet val   : {len(val_tiles):>5} tiles  "
          f"from {len(val_sheets)} sheets  (≠ '{sheet_id}')")

    if not fb_tiles:
        sys.exit(
            f"No feedback tiles found for sheet '{sheet_id}'.\n"
            f"Run rasterise.py first:\n"
            f"  conda activate maptools\n"
            f"  python steps/06_feedback/lines/rasterise.py --sheet {sheet_id}"
        )
    if not val_tiles:
        print(
            "Warning: no cross-sheet validation tiles found. "
            "Early stopping will use training loss instead.\n"
            "Process at least one other sheet to enable cross-sheet validation."
        )

    # ---- Build mixed training arrays ----------------------------------------
    print(f"\nBuilding training mix  "
          f"(replay_ratio={replay_ratio:.2f}  fb_weight={fb_weight:.2f})...")
    X_train, y_train, w_train = build_training_arrays(
        gt_tiles, fb_tiles, replay_ratio, fb_weight,
    )
    n_gt_used = int((w_train == 1.0).sum())
    n_fb_used = int((w_train  < 1.0).sum())
    print(f"  Training mix : {n_gt_used} GT tiles + {n_fb_used} feedback tiles "
          f"= {len(X_train)} total")
    actual_ratio = n_gt_used / len(X_train) if len(X_train) > 0 else 0
    print(f"  Actual replay ratio : {actual_ratio:.2f}  (target {replay_ratio:.2f})")

    # ---- Build model --------------------------------------------------------
    model = build_model(tile_size, unet_cfg["image_channels"], loss_type)

    # ---- Load starting weights ----------------------------------------------
    # Search order:
    #   1. --weights CLI argument (explicit path)
    #   2. Most recently modified feedback_v* in iterative/
    #   3. Most recently modified *.weights.h5 in models/base/ (recursive)
    # Working weights (step 03) are intentionally excluded — feedback always
    # starts from the general iterative model, not a sheet-specific finetune.
    start_weights = args.weights
    if start_weights is None:
        candidates = sorted(
            iterative_dir.glob("feedback_v*_best.weights.h5"),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            start_weights = str(candidates[-1])
    if start_weights is None:
        base_candidates = sorted(
            (ROOT / paths["models_base"]).rglob("*.weights.h5"),
            key=lambda p: p.stat().st_mtime,
        )
        if base_candidates:
            start_weights = str(base_candidates[-1])

    if start_weights and Path(start_weights).exists():
        p = Path(start_weights)
        source = "iterative" if "iterative" in p.parts else "base"
        model.load_weights(start_weights)
        print(f"\nStarting weights [{source}]: {p.name}")
    else:
        print("\nNo starting weights found — fine-tuning from random initialisation")
        print("  Pass --weights <path> to specify a weights file.")

    # ---- Freeze encoder blocks ----------------------------------------------
    if freeze_n > 0:
        n_frozen = freeze_encoder_blocks(model, freeze_n)
        print(f"Encoder blocks 1–{freeze_n} frozen: {n_frozen} layers set non-trainable")
    else:
        print("Encoder freezing disabled (--no-freeze)")

    # Recompile at feedback LR (freeze + new LR both require recompile)
    model.compile(
        optimizer=Adam(learning_rate=fb_lr),
        loss=LOSS_MAP[loss_type],
        metrics=[dsc, clDice, tp, tn, prec, recall],
    )

    trainable_params = sum(
        tf.size(v).numpy() for v in model.trainable_variables
    )
    total_params = sum(tf.size(v).numpy() for v in model.variables)
    print(f"Trainable params: {trainable_params:,} / {total_params:,}  "
          f"({100 * trainable_params / total_params:.1f}%)")

    # ---- Build tf.data.Dataset ----------------------------------------------
    train_ds = make_weighted_augmented_dataset(
        X_train, y_train, w_train, batch_size, tile_size,
    )

    # ---- Callbacks ----------------------------------------------------------
    best_path = iterative_dir / f"{run_name}_best.weights.h5"
    log_path  = logs_dir      / f"{run_name}_metrics.csv"

    if val_tiles:
        cb = CrossSheetCallback(
            val_tiles         = val_tiles,
            tau               = apl_tau,
            threshold         = threshold,
            best_weights_path = best_path,
            log_path          = log_path,
            max_val_tiles     = max_val_tiles,
            patience          = patience,
        )
        callbacks = [cb]
    else:
        # No cross-sheet val — fall back to saving best by training loss
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                str(best_path), save_weights_only=True,
                monitor="loss", save_best_only=True, verbose=1,
            ),
        ]
        # Write a minimal log header so downstream tools don't break
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss"])

    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1,
        )
    )

    # ---- Train --------------------------------------------------------------
    print(f"\n── Training ────────────────────────────────────────────────")
    print(f"Run name     : {run_name}")
    print(f"Epochs       : {epochs}  |  LR: {fb_lr}  |  Batch: {batch_size}")
    print(f"Best weights → {best_path.relative_to(ROOT)}")
    print(f"Metrics log  → {log_path.relative_to(ROOT)}\n")

    model.fit(
        train_ds,
        epochs    = epochs,
        callbacks = callbacks,
        verbose   = 1,
    )

    best_f1 = callbacks[0].best_f1 if val_tiles else float("nan")
    print(f"\nDone.")
    if val_tiles:
        print(f"Best cross-sheet path_f1 = {best_f1:.4f}")
    print(f"Weights saved → {best_path.relative_to(ROOT)}")
    print(
        f"\nThese weights are now the active iterative model.\n"
        f"Step 03 finetune and step 04 predict will use them automatically\n"
        f"as the starting point for the next sheet.\n"
        f"\nTo compare iterative versions run:\n"
        f"  python steps/03_finetune/lines/evaluate.py"
    )


if __name__ == "__main__":
    main()
