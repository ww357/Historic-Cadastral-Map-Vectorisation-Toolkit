"""
Compare boundary U-Net weight files on an annotated evaluation set.

Usage
-----
    python evaluate.py                              # all weights in models/finetuned/ + models/base/
    python evaluate.py --weights-dir models/base/   # specific directory
    python evaluate.py --data-dir data/annotations/ # override annotation directory

Output
------
  Terminal table sorted by path_f1 (best first)
  models/finetuned/evaluation_results.csv

Note on Est. Mending Time
  Uses log-linear fit: mending_time = 0.3421 * APL^0.647   R²=0.317
  Reliable for ranking models against each other; too uncertain for scheduling.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.unet.architecture import build_model
from models.unet.metrics import compute_path_metrics, MENDING_C, MENDING_K

import tensorflow as tf


# ---------------------------------------------------------------------------
# Data loading  (shared logic with train.py)
# ---------------------------------------------------------------------------

def load_eval_tiles(patches_dir: Path, masks_dir: Path, tile_size: int):
    tiles = []
    for patch_path in sorted(patches_dir.glob("*.png")):
        mask_path = masks_dir / patch_path.name
        if not mask_path.exists():
            continue
        img  = np.array(Image.open(patch_path).convert("L"), dtype=np.float32) / 255.0
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = (mask > 127).astype(np.float32)
        h, w = img.shape
        for r in range(h // tile_size):
            for c in range(w // tile_size):
                rs, re = r * tile_size, (r + 1) * tile_size
                cs, ce = c * tile_size, (c + 1) * tile_size
                tiles.append((img[rs:re, cs:ce], mask[rs:re, cs:ce]))
    return tiles


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_weights(model, weight_path: Path, tiles, threshold: float, tau: int) -> dict:
    model.load_weights(str(weight_path))
    per_tile = []

    for img_tile, gt_tile in tiles:
        inp  = img_tile[np.newaxis, ..., np.newaxis]
        pred = model.predict(inp, verbose=0)[0, ..., 0]
        m = compute_path_metrics(
            pred_mask = (pred >= threshold).astype(np.uint8),
            gt_mask   = (gt_tile >= 0.5).astype(np.uint8),
            tau       = tau,
        )
        per_tile.append(m)

    apl_total = sum(m["apl"]       for m in per_tile)
    fn_total  = sum(m["fn_length"] for m in per_tile)
    fp_total  = sum(m["fp_length"] for m in per_tile)
    est_time  = MENDING_C * (apl_total ** MENDING_K) if apl_total > 0 else 0.0

    return dict(
        weights          = weight_path.name,
        path_f1          = round(float(np.mean([m["path_f1"]        for m in per_tile])), 4),
        path_recall      = round(float(np.mean([m["path_recall"]    for m in per_tile])), 4),
        path_precision   = round(float(np.mean([m["path_precision"] for m in per_tile])), 4),
        apl_total_px     = int(round(apl_total)),
        fn_total_px      = int(round(fn_total)),
        fp_total_px      = int(round(fp_total)),
        est_mending_time = round(est_time, 2),
    )


def print_table(results: list[dict]):
    cols  = ["weights", "path_f1", "path_recall", "path_precision",
             "apl_total_px", "fn_total_px", "fp_total_px", "est_mending_time"]
    heads = ["Weights", "F1", "Recall", "Prec",
             "APL (px)", "FN (px)", "FP (px)", "Est.Time (min)"]
    widths = [38, 7, 8, 7, 10, 9, 9, 15]

    sep = "  ".join("-" * w for w in widths)
    hdr = "  ".join(h.ljust(w) for h, w in zip(heads, widths))
    print("\n" + sep)
    print(hdr)
    print(sep)
    for r in results:
        row = [str(r[c]) for c in cols]
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))
    print(sep)
    print("\nNote: Est.Time uses log-linear fit (R²=0.317) — use for model ranking only.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare boundary U-Net weight files")
    parser.add_argument("--config",      default="config.yaml")
    parser.add_argument("--weights-dir", default=None,
                        help="Directory to scan for .weights.h5 files")
    parser.add_argument("--data-dir",    default=None,
                        help="Annotations root (overrides paths.annotations in config)")
    args = parser.parse_args()

    cfg       = yaml.safe_load((ROOT / args.config).read_text())
    unet_cfg  = cfg["unet"]
    ft_cfg    = cfg["finetune"]
    paths_cfg = cfg["paths"]

    tile_size   = unet_cfg["inference_size"]
    patches_dir = ROOT / paths_cfg["patches"]
    masks_dir   = ROOT / (args.data_dir or paths_cfg["annotations"]) / "boundaries" / "masks"

    # Collect weight files from base + finetuned dirs (or override)
    if args.weights_dir:
        search_dirs = [ROOT / args.weights_dir]
    else:
        search_dirs = [
            ROOT / paths_cfg["models_base"],
            ROOT / paths_cfg["models_finetuned"],
        ]

    weight_files = []
    for d in search_dirs:
        if d.exists():
            weight_files += sorted(d.glob("*.weights.h5"))
            weight_files += sorted(d.glob("*.h5"))

    if not weight_files:
        sys.exit(f"No weight files found in: {search_dirs}")

    # Load eval tiles
    print("Loading evaluation tiles...")
    tiles = load_eval_tiles(patches_dir, masks_dir, tile_size)
    if not tiles:
        sys.exit(
            f"No paired eval tiles found.\n"
            f"  Patches: {patches_dir}\n"
            f"  Masks:   {masks_dir}"
        )
    print(f"  {len(tiles)} tiles\n")

    model = build_model(
        inference_size = tile_size,
        channels       = unet_cfg["image_channels"],
        loss_type      = unet_cfg["loss_type"],
    )

    results = []
    for wf in weight_files:
        print(f"  Evaluating {wf.name}...")
        try:
            r = evaluate_weights(model, wf, tiles,
                                 threshold = unet_cfg["threshold"],
                                 tau       = ft_cfg["apl_tau"])
            results.append(r)
        except Exception as e:
            print(f"    ✗ Skipped: {e}")

    if not results:
        sys.exit("All weight files failed evaluation.")

    results.sort(key=lambda x: x["path_f1"], reverse=True)
    print_table(results)

    # Save CSV alongside the finetuned weights
    out_dir = ROOT / paths_cfg["models_finetuned"]
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "evaluation_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
