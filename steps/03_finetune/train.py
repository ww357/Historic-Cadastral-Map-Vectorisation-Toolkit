"""
Fine-tune the boundary U-Net on labelme-annotated patches.

Usage
-----
    python train.py                               # uses config.yaml defaults
    python train.py --weights base.weights.h5     # fine-tune from specific weights
    python train.py --name finetune_v1            # tag the run

Data expected
-------------
    data/patches/           PNG patches from step 01_patchify
    data/annotations/boundaries/   binary mask PNGs from step 02_annotate/export_masks.py
    File names must match (e.g. patch_0000.png in both folders).

Outputs
-------
    models/finetuned/{name}_best.weights.h5   — best weights by path_f1
    models/logs/{name}_metrics.csv            — per-epoch metrics
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.unet.architecture import build_model, LOSS_MAP, dsc, clDice, tp, tn, prec, recall
from models.unet.augmentation import make_augmented_dataset
from models.unet.metrics import compute_path_metrics, MENDING_C, MENDING_K

import tensorflow as tf
from tensorflow.keras.optimizers import Adam


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_patch_tiles(patches_dir: Path, masks_dir: Path, tile_size: int):
    """
    Load paired 512px patches + masks and split each into tile_size tiles.
    Returns a list of dicts: {patch_name, img_tile, mask_tile}
    Patch-level grouping is preserved so train/val split avoids leakage.
    """
    records = []
    for patch_path in sorted(patches_dir.glob("*.png")):
        mask_path = masks_dir / patch_path.name
        if not mask_path.exists():
            continue

        img  = np.array(Image.open(patch_path).convert("L"),  dtype=np.float32) / 255.0
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = (mask > 127).astype(np.float32)

        h, w = img.shape
        n_rows, n_cols = h // tile_size, w // tile_size
        for r in range(n_rows):
            for c in range(n_cols):
                rs, re = r * tile_size, (r + 1) * tile_size
                cs, ce = c * tile_size, (c + 1) * tile_size
                records.append(dict(
                    patch_name = patch_path.stem,
                    img_tile   = img [rs:re, cs:ce],
                    mask_tile  = mask[rs:re, cs:ce],
                ))

    return records


# ---------------------------------------------------------------------------
# Callback: compute APL / path-F1 on validation set after each epoch
# ---------------------------------------------------------------------------

class PathMetricsCallback(tf.keras.callbacks.Callback):
    """
    After each epoch:
      - Runs inference on validation tiles
      - Computes APL, path-recall, path-precision, path-F1
      - Saves weights if path_f1 improves
      - Appends a row to the metrics CSV
    """

    def __init__(self, val_tiles, tau, threshold, best_weights_path, log_path,
                 max_val_tiles=200, early_stopping_patience=10):
        super().__init__()
        # Cap val tiles for speed; shuffle so the cap isn't always the same patches
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(val_tiles))[:max_val_tiles]
        self.val_tiles          = [val_tiles[i] for i in idx]
        self.tau                = tau
        self.threshold          = threshold
        self.best_weights_path  = best_weights_path
        self.log_path           = log_path
        self.best_f1            = -1.0
        self.patience           = early_stopping_patience
        self._no_improve        = 0
        self._init_log()

    def _init_log(self):
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "loss",
                "apl_total", "fn_total", "fp_total",
                "path_recall", "path_precision", "path_f1",
                "est_mending_time",
            ])

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        per_tile = []

        for rec in self.val_tiles:
            inp  = rec["img_tile"][np.newaxis, ..., np.newaxis]
            pred = self.model.predict(inp, verbose=0)[0, ..., 0]
            m = compute_path_metrics(
                pred_mask = (pred >= self.threshold).astype(np.uint8),
                gt_mask   = (rec["mask_tile"] >= 0.5).astype(np.uint8),
                tau       = self.tau,
            )
            per_tile.append(m)

        # Aggregate — F1/recall/precision as mean; APL as sum (additive)
        path_f1   = float(np.mean([m["path_f1"]        for m in per_tile]))
        path_rec  = float(np.mean([m["path_recall"]    for m in per_tile]))
        path_prec = float(np.mean([m["path_precision"] for m in per_tile]))
        apl_total = sum(m["apl"]       for m in per_tile)
        fn_total  = sum(m["fn_length"] for m in per_tile)
        fp_total  = sum(m["fp_length"] for m in per_tile)
        # Apply log-linear to total APL (regression was fit on map-level, not tile-level)
        est_time  = MENDING_C * (apl_total ** MENDING_K) if apl_total > 0 else 0.0

        print(f"\n  Path-F1: {path_f1:.4f}  "
              f"(R={path_rec:.3f}  P={path_prec:.3f})  "
              f"APL: {apl_total:.0f}px  "
              f"Est. mending: {est_time:.1f} min  (relative only, R²=0.317)")

        # Checkpoint + early stopping
        if path_f1 > self.best_f1:
            self.best_f1    = path_f1
            self._no_improve = 0
            self.model.save_weights(str(self.best_weights_path))
            print(f"  ✓ New best path_f1={path_f1:.4f} — weights saved")
        else:
            self._no_improve += 1
            print(f"  No improvement ({self._no_improve}/{self.patience})")
            if self._no_improve >= self.patience:
                print(f"  Early stopping triggered — best path_f1={self.best_f1:.4f}")
                self.model.stop_training = True

        # Log
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1,
                round(logs.get("loss", 0), 6),
                round(apl_total, 1), round(fn_total, 1), round(fp_total, 1),
                round(path_rec,  4), round(path_prec, 4), round(path_f1,  4),
                round(est_time,  2),
            ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune boundary U-Net")
    parser.add_argument("--config",  default="config.yaml",
                        help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--weights", default=None,
                        help="Base weights file to fine-tune from")
    parser.add_argument("--name",    default=None,
                        help="Run name for checkpoint and log files")
    parser.add_argument("--sheet",   default=None,
                        help="Map sheet name (subdirectory under patches/images/ and annotations/)")
    args = parser.parse_args()

    cfg          = yaml.safe_load((ROOT / args.config).read_text())
    unet_cfg     = cfg["unet"]
    ft_cfg       = cfg["finetune"]
    paths_cfg    = cfg["paths"]

    patches_base = ROOT / paths_cfg["patches"] / "images"
    patches_dir  = patches_base / args.sheet if args.sheet else patches_base
    masks_base   = ROOT / paths_cfg["annotations"] / "boundaries"
    masks_dir    = (masks_base / args.sheet / "masks") if args.sheet else masks_base

    # Auto-export masks from labelme JSON if not yet done
    if not masks_dir.exists() or not any(masks_dir.glob("*.png")):
        print(f"Masks not found at {masks_dir}")
        print("Running export_masks.py first...\n")
        import subprocess
        result = subprocess.run(
            [sys.executable,
             str(ROOT / "steps" / "02_annotate" / "export_masks.py"),
             "--sheet", args.sheet],
            check=False,
        )
        if result.returncode != 0:
            sys.exit("export_masks.py failed — check your annotations and try again.")
        print()
    weights_dir  = ROOT / paths_cfg["models_finetuned"]
    logs_dir     = ROOT / paths_cfg["logs"]
    weights_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.name or datetime.now().strftime("finetune_%Y%m%d_%H%M")
    tile_size = unet_cfg["inference_size"]

    # ---- Load data ---------------------------------------------------------
    print("Loading patches...")
    all_records = load_patch_tiles(patches_dir, masks_dir, tile_size)
    if not all_records:
        sys.exit(
            f"No paired patches found.\n"
            f"  Patches: {patches_dir}\n"
            f"  Masks:   {masks_dir}\n"
            + ("" if args.sheet else "Tip: use --sheet <name> to target a specific map sheet.\n")
            + "Check paths.patches and paths.annotations in config.yaml."
        )

    # Split by patch name — all tiles from a patch stay in the same split
    patch_names = sorted({r["patch_name"] for r in all_records})
    rng = np.random.default_rng(42)
    rng.shuffle(patch_names)
    n_val = max(1, int(len(patch_names) * ft_cfg["val_split"]))
    val_set   = set(patch_names[:n_val])
    train_set = set(patch_names[n_val:])

    train_records = [r for r in all_records if r["patch_name"] in train_set]
    val_records   = [r for r in all_records if r["patch_name"] in val_set]

    print(f"  {len(patch_names)} patches → "
          f"{len(train_records)} train tiles / {len(val_records)} val tiles")

    X_train = np.stack([r["img_tile"]  for r in train_records])[..., np.newaxis]
    y_train = np.stack([r["mask_tile"] for r in train_records])[..., np.newaxis]

    # ---- Build model -------------------------------------------------------
    model = build_model(
        inference_size = tile_size,
        channels       = unet_cfg["image_channels"],
        loss_type      = unet_cfg["loss_type"],
    )

    # Recompile at fine-tuning LR (build_model hardcodes 1e-3; recompile is version-safe)
    model.compile(
        optimizer = Adam(learning_rate=ft_cfg["learning_rate"]),
        loss      = LOSS_MAP[unet_cfg["loss_type"]],
        metrics   = [dsc, clDice, tp, tn, prec, recall],
    )

    # Load base weights — search order:
    #   1. --weights CLI argument (explicit path)
    #   2. Most recent file in models/base/
    #   3. models/finetuned/model_weights.weights.h5
    base_weights = args.weights
    if base_weights is None:
        candidates = sorted((ROOT / paths_cfg["models_base"]).rglob("*.weights.h5"))
        if candidates:
            base_weights = str(candidates[-1])
    if base_weights is None:
        fallback = ROOT / paths_cfg["models_finetuned"] / "model_weights.weights.h5"
        if fallback.exists():
            base_weights = str(fallback)

    if base_weights and Path(base_weights).exists():
        model.load_weights(base_weights)
        print(f"Loaded base weights: {base_weights}")
    else:
        print("No base weights found — training from scratch")
        print(f"  Searched: {ROOT / paths_cfg['models_base']}")
        print(f"  Searched: {ROOT / paths_cfg['models_finetuned'] / 'model_weights.weights.h5'}")
        print("  Pass --weights <path> to specify a weights file explicitly.")

    # ---- Dataset -----------------------------------------------------------
    train_ds = make_augmented_dataset(
        X          = X_train,
        y          = y_train,
        batch_size = unet_cfg["batch_size"],
        patch_size = tile_size,
    )

    # ---- Callbacks ---------------------------------------------------------
    best_path = weights_dir / f"{run_name}_best.weights.h5"
    log_path  = logs_dir    / f"{run_name}_metrics.csv"

    callbacks = [
        PathMetricsCallback(
            val_tiles                = val_records,
            tau                      = ft_cfg["apl_tau"],
            threshold                = unet_cfg["threshold"],
            best_weights_path        = best_path,
            log_path                 = log_path,
            max_val_tiles            = ft_cfg.get("max_val_tiles", 200),
            early_stopping_patience  = ft_cfg.get("early_stopping_patience", 10),
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1,
        ),
    ]

    # ---- Train -------------------------------------------------------------
    print(f"\nRun: {run_name}")
    print(f"Best weights → {best_path}")
    print(f"Metrics log  → {log_path}\n")

    model.fit(
        train_ds,
        epochs    = ft_cfg["epochs"],
        callbacks = callbacks,
        verbose   = 1,
    )

    print(f"\nDone. Best path_f1={callbacks[0].best_f1:.4f}")


if __name__ == "__main__":
    main()
