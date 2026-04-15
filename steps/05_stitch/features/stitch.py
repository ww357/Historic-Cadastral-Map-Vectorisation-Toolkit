"""
Stitch MapSAM feature prediction masks into a full-document binary GeoTIFF.

Any feature label used during annotation/prediction can be stitched — there is
no fixed feature list.  Pass --feature with the same label name used in labelme.

Reads  : data/patches/metadata/<SHEET_ID>_patches.csv       — patch offsets + georef
         data/predictions/<FEATURE>/<SHEET_ID>/*.png        — 512px binary masks
         data/annotations/<FEATURE>/<SHEET_ID>/masks/*.png  — manual annotation masks (fallback)
         data/raw/<SHEET_ID>/<SHEET_ID>.tif                 — source of true dimensions + CRS

Writes : data/stitched/<FEATURE>/<SHEET_ID>.tif             — full-document uint8 GeoTIFF

For patches that were annotated manually, the annotation mask is used directly
instead of a model prediction — these patches were skipped during 04_predict.
This produces a cleaner result in annotated areas, reducing mending time.

Each prediction PNG is 512px but edge patches may have a smaller valid region
(patch_w × patch_h from the metadata). Only the valid region is placed onto the
canvas; the white-padded remainder is discarded.

For overlapping patches (overlap > 0 in config): pixels covered by multiple
patches take the maximum prediction value — union of all predictions.

Usage:
    python stitch.py --sheet Timberscombe --feature water
    python stitch.py --sheet Timberscombe --feature building
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from PIL import Image
from rasterio.transform import Affine
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]


def load_config() -> dict:
    p = ROOT / "config.yaml"
    if not p.exists():
        sys.exit(f"config.yaml not found at {p}")
    return yaml.safe_load(p.read_text())


def stitch(sheet_id: str, feature: str, repo_root: Path):
    cfg = load_config()

    raw_path  = repo_root / cfg["paths"]["raw"]         / sheet_id / f"{sheet_id}.tif"
    meta_path = repo_root / cfg["paths"]["patches"]     / "metadata" / f"{sheet_id}_patches.csv"
    pred_dir  = repo_root / cfg["paths"]["predictions"] / feature / sheet_id
    out_dir   = repo_root / cfg["paths"]["stitched"]    / feature
    out_path  = out_dir / f"{sheet_id}.tif"

    if not meta_path.exists():
        sys.exit(f"Metadata CSV not found: {meta_path}")
    if not pred_dir.exists():
        print(f"Warning: predictions dir not found: {pred_dir}")
        print("  Annotation masks will be used where available; other patches will be blank.")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Annotation masks for patches that were manually annotated — used directly
    # instead of model predictions to preserve ground-truth quality in those areas.
    ann_mask_dir = repo_root / cfg["paths"]["annotations"] / feature / sheet_id / "masks"

    # --- Source dimensions and georef ---
    if raw_path.exists():
        with rasterio.open(raw_path) as src:
            img_w, img_h = src.width, src.height
            has_georef   = src.crs is not None
            crs          = src.crs if has_georef else None
            transform    = src.transform
    else:
        print(f"Warning: raw file not found at {raw_path}, deriving dimensions from metadata.")
        meta_tmp   = pd.read_csv(meta_path)
        img_w      = int((meta_tmp["col_off"] + meta_tmp["patch_w"]).max())
        img_h      = int((meta_tmp["row_off"] + meta_tmp["patch_h"]).max())
        has_georef = bool(meta_tmp["has_georef"].iloc[0])
        if has_georef:
            r0        = meta_tmp[(meta_tmp["row_off"] == 0) & (meta_tmp["col_off"] == 0)].iloc[0]
            transform = Affine(r0.tf_a, r0.tf_b, r0.tf_c, r0.tf_d, r0.tf_e, r0.tf_f)
            crs       = meta_tmp["crs"].iloc[0]
        else:
            transform, crs = None, None

    print(f"Sheet      : {sheet_id}")
    print(f"Feature    : {feature}")
    print(f"Canvas     : {img_w} × {img_h} px")
    print(f"CRS        : {crs or 'none'}")

    # --- Load metadata ---
    meta = pd.read_csv(meta_path)
    print(f"Patches    : {len(meta)} in metadata")

    # --- Build canvas ---
    canvas = np.zeros((img_h, img_w), dtype=np.uint8)

    missing, from_pred, from_ann = 0, 0, 0
    for _, row in tqdm(meta.iterrows(), total=len(meta), unit="patch"):
        pred_path = pred_dir / f"{row.patch_id}.png"
        ann_path  = ann_mask_dir / f"{row.patch_id}.png"

        if pred_path.exists():
            pred = np.array(Image.open(pred_path).convert("L"))
            from_pred += 1
        elif ann_path.exists():
            # No model prediction for this patch — use the manual annotation directly
            pred = np.array(Image.open(ann_path).convert("L"))
            from_ann += 1
        else:
            missing += 1
            continue

        ph = int(row.patch_h)
        pw = int(row.patch_w)
        r  = int(row.row_off)
        c  = int(row.col_off)

        # np.maximum: union of all predictions for overlapping patches
        canvas[r:r + ph, c:c + pw] = np.maximum(
            canvas[r:r + ph, c:c + pw],
            pred[:ph, :pw],
        )

    placed = from_pred + from_ann
    print(f"\nPlaced {placed} patches  "
          f"({from_pred} predicted, {from_ann} from annotations, {missing} missing)")

    # --- Save ---
    profile = {
        "driver":   "GTiff",
        "dtype":    "uint8",
        "width":    img_w,
        "height":   img_h,
        "count":    1,
        "compress": "lzw",
    }
    if has_georef and crs is not None:
        profile["crs"]       = crs
        profile["transform"] = transform

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(canvas[np.newaxis, :, :])

    n_feature_px = int((canvas > 0).sum())
    print(f"Saved → {out_path.relative_to(repo_root)}")
    print(f"  {feature} pixels: {n_feature_px:,}  "
          f"({100 * n_feature_px / (img_w * img_h):.2f}% of image)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stitch MapSAM feature predictions into a full-document GeoTIFF."
    )
    parser.add_argument("--sheet",   required=True, help="Sheet ID")
    parser.add_argument("--feature", required=True,
                        help="Feature class — any label used in labelme annotations "
                             "(e.g. water, building, vegetation)")
    args = parser.parse_args()
    stitch(args.sheet, args.feature, ROOT)
