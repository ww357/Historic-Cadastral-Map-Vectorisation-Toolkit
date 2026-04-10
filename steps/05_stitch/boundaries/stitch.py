"""
Stitch boundary prediction masks into a full-document binary GeoTIFF.

Reads  : data/patches/metadata/<SHEET_ID>_patches.csv   — patch offsets + georef
         data/predictions/boundaries/<SHEET_ID>/*.png   — 1024px binary masks
         data/raw/<SHEET_ID>/<SHEET_ID>.tif             — source of true dimensions + CRS

Writes : data/stitched/boundaries/<SHEET_ID>.tif        — full-document uint8 GeoTIFF

Each prediction PNG is 1024px but edge patches may have a smaller valid region
(patch_w × patch_h from the metadata). Only the valid region is placed onto the
canvas; the white-padded remainder is discarded.

For overlapping patches (overlap > 0 in config): pixels covered by multiple
patches take the maximum prediction value — if any patch predicts a boundary
at that pixel, it is marked as boundary in the output.

Usage:
    python stitch.py --sheet SHEET_ID
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]


def load_config() -> dict:
    p = ROOT / "config.yaml"
    if not p.exists():
        sys.exit(f"config.yaml not found at {p}")
    return yaml.safe_load(p.read_text())


def stitch(sheet_id: str, repo_root: Path):
    cfg = load_config()

    raw_path   = repo_root / cfg["paths"]["raw"]         / sheet_id / f"{sheet_id}.tif"
    meta_path  = repo_root / cfg["paths"]["patches"]     / "metadata" / f"{sheet_id}_patches.csv"
    pred_dir   = repo_root / cfg["paths"]["predictions"] / "boundaries" / sheet_id
    out_dir    = repo_root / cfg["paths"]["stitched"]    / "boundaries"
    out_path   = out_dir / f"{sheet_id}.tif"

    for p, label in [(meta_path, "Metadata CSV"), (pred_dir, "Predictions dir")]:
        if not p.exists():
            sys.exit(f"{label} not found: {p}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Source dimensions and georef ---
    if raw_path.exists():
        with rasterio.open(raw_path) as src:
            img_w, img_h = src.width, src.height
            has_georef   = src.crs is not None
            crs          = src.crs if has_georef else None
            transform    = src.transform
    else:
        # Derive dimensions from metadata if raw file unavailable
        print(f"Warning: raw file not found at {raw_path}, deriving dimensions from metadata.")
        meta_tmp  = pd.read_csv(meta_path)
        img_w     = int((meta_tmp["col_off"] + meta_tmp["patch_w"]).max())
        img_h     = int((meta_tmp["row_off"] + meta_tmp["patch_h"]).max())
        has_georef = meta_tmp["has_georef"].iloc[0]
        if has_georef:
            r0        = meta_tmp.loc[meta_tmp["row_off"].idxmin()]
            r0        = meta_tmp[(meta_tmp["row_off"] == 0) & (meta_tmp["col_off"] == 0)].iloc[0]
            transform = Affine(r0.tf_a, r0.tf_b, r0.tf_c, r0.tf_d, r0.tf_e, r0.tf_f)
            crs       = meta_tmp["crs"].iloc[0]
        else:
            transform, crs = None, None

    print(f"Sheet      : {sheet_id}")
    print(f"Canvas     : {img_w} × {img_h} px")
    print(f"CRS        : {crs or 'none'}")

    # --- Load metadata ---
    meta = pd.read_csv(meta_path)
    print(f"Patches    : {len(meta)} in metadata")

    # --- Build canvas ---
    canvas = np.zeros((img_h, img_w), dtype=np.uint8)

    missing, placed = 0, 0
    for _, row in tqdm(meta.iterrows(), total=len(meta), unit="patch"):
        pred_path = pred_dir / f"{row.patch_id}.png"
        if not pred_path.exists():
            missing += 1
            continue

        pred = np.array(Image.open(pred_path).convert("L"))

        # Only the valid (unpadded) region — discard edge-patch padding
        ph = int(row.patch_h)
        pw = int(row.patch_w)
        r  = int(row.row_off)
        c  = int(row.col_off)

        # np.maximum handles overlapping patches — union of all predictions
        canvas[r:r + ph, c:c + pw] = np.maximum(
            canvas[r:r + ph, c:c + pw],
            pred[:ph, :pw]
        )
        placed += 1

    print(f"\nPlaced {placed} patches  ({missing} missing predictions skipped)")

    # --- Save ---
    profile = {
        "driver": "GTiff",
        "dtype":  "uint8",
        "width":  img_w,
        "height": img_h,
        "count":  1,
        "compress": "lzw",
    }
    if has_georef and crs is not None:
        profile["crs"]       = crs
        profile["transform"] = transform

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(canvas[np.newaxis, :, :])  # rasterio expects (bands, H, W)

    print(f"Saved → {out_path.relative_to(repo_root)}")
    print(f"  Boundary pixels: {(canvas > 0).sum():,}  "
          f"({100 * (canvas > 0).mean():.2f}% of image)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stitch boundary predictions into a full-document GeoTIFF."
    )
    parser.add_argument("--sheet", required=True, help="Sheet ID")
    args = parser.parse_args()
    stitch(args.sheet, ROOT)
