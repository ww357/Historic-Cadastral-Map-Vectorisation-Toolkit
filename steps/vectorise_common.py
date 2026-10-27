"""
Shared full-sheet stitching for the vectorise steps.

Reassembles 512px prediction patches (falling back to annotation masks where a
patch was hand-labelled) onto a full-sheet canvas, writes a stitched GeoTIFF, and
returns the georeferencing the vectorise stage needs. Parameterised by the
feature's prediction / annotation / stitched folder names, so the lines, dashed,
and MapSAM-polygon steps all share one implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.transform import Affine
from tqdm import tqdm

from common import ROOT, find_raw, rel_to_root


def georef_from_metadata(meta: pd.DataFrame) -> dict:
    """
    Whole-sheet georeferencing recovered from the patch metadata CSV (used when the raw map
    is not on disk). Each row stores that PATCH's transform, so the sheet's origin is the
    patch origin minus the patch's pixel offset - this works from any patch, not just the
    top-left one (which usually isn't saved: patches outside the map-area mask are skipped).
    """
    has_georef = bool(meta["has_georef"].iloc[0])
    if not has_georef:
        return {"transform": None, "crs": None, "has_georef": False}
    r = meta.iloc[0]
    return {
        "transform": Affine(r.tf_a, r.tf_b, r.tf_c - r.col_off * r.tf_a - r.row_off * r.tf_b,
                            r.tf_d, r.tf_e, r.tf_f - r.col_off * r.tf_d - r.row_off * r.tf_e),
        "crs": meta["crs"].iloc[0],
        "has_georef": True,
    }


def sheet_georef(sheet_id: str, cfg: dict) -> dict:
    """{"transform", "crs", "has_georef", "width", "height"} for a sheet: from the raw map's
    header when it is on disk, otherwise from the patch metadata CSV."""
    paths = cfg["paths"]
    raw_path = find_raw(ROOT / paths["raw"], sheet_id)
    if raw_path is not None:
        with rasterio.open(raw_path) as src:
            return {"transform": src.transform if src.crs else None, "crs": src.crs,
                    "has_georef": src.crs is not None, "width": src.width, "height": src.height}
    meta_path = ROOT / paths["patches"] / "metadata" / f"{sheet_id}_patches.csv"
    if not meta_path.exists():
        sys.exit(f"Neither the raw map nor {rel_to_root(meta_path)} was found - run patchify first.")
    meta = pd.read_csv(meta_path)
    g = georef_from_metadata(meta)
    g["width"] = int((meta["col_off"] + meta["patch_w"]).max())
    g["height"] = int((meta["row_off"] + meta["patch_h"]).max())
    return g


def stitch(sheet_id: str, cfg: dict, pred_name: str, ann_name: str,
           stitched_name: str, pixel_label: str) -> tuple[Path, dict]:
    """
    Reassemble patch predictions (+ annotation fallbacks) into a full-sheet GeoTIFF.

    pred_name     : predictions/<pred_name>/<sheet>/       prediction patch folder
    ann_name      : annotations/<ann_name>/<sheet>/masks/  annotation fallback masks
    stitched_name : stitched/<stitched_name>/<sheet>.tif   output raster
    pixel_label   : label used in the "<label> pixels" summary line

    Returns (stitched_path, {"transform", "crs", "has_georef"}).
    """
    paths = cfg["paths"]
    raw_path     = find_raw(ROOT / paths["raw"], sheet_id)
    meta_path    = ROOT / paths["patches"]     / "metadata" / f"{sheet_id}_patches.csv"
    pred_dir     = ROOT / paths["predictions"] / pred_name / sheet_id
    ann_mask_dir = ROOT / paths["annotations"] / ann_name / sheet_id / "masks"
    out_dir      = ROOT / paths["stitched"]    / stitched_name
    out_path     = out_dir / f"{sheet_id}.tif"

    if not meta_path.exists():
        sys.exit(f"Metadata CSV not found: {meta_path}")
    if not pred_dir.exists():
        print(f"Warning: predictions dir not found: {pred_dir}")
        print("  Annotation masks will be used where available; other patches will be blank.")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Source dimensions + georef - a cheap header read (any GDAL format); fall back
    # to the metadata CSV, which records the same values, when the raw map is absent.
    if raw_path is not None:
        with rasterio.open(raw_path) as src:
            img_w, img_h = src.width, src.height
            has_georef   = src.crs is not None
            crs          = src.crs if has_georef else None
            transform    = src.transform
    else:
        print("Warning: raw map not found, deriving dimensions from metadata.")
        meta_tmp   = pd.read_csv(meta_path)
        img_w      = int((meta_tmp["col_off"] + meta_tmp["patch_w"]).max())
        img_h      = int((meta_tmp["row_off"] + meta_tmp["patch_h"]).max())
        g = georef_from_metadata(meta_tmp)
        has_georef, transform, crs = g["has_georef"], g["transform"], g["crs"]

    print(f"\n-- Stitch --------------------------------------")
    print(f"Sheet      : {sheet_id}")
    print(f"Canvas     : {img_w} x {img_h} px  |  CRS: {crs or 'none'}")

    meta   = pd.read_csv(meta_path)
    canvas = np.zeros((img_h, img_w), dtype=np.uint8)

    missing, from_pred, from_ann = 0, 0, 0
    for _, row in tqdm(meta.iterrows(), total=len(meta), unit="patch"):
        pred_path = pred_dir     / f"{row.patch_id}.png"
        ann_path  = ann_mask_dir / f"{row.patch_id}.png"

        if pred_path.exists():
            pred = np.array(Image.open(pred_path).convert("L"))
            from_pred += 1
        elif ann_path.exists():
            pred = np.array(Image.open(ann_path).convert("L"))
            from_ann += 1
        else:
            missing += 1
            continue

        ph, pw = int(row.patch_h), int(row.patch_w)
        r,  c  = int(row.row_off), int(row.col_off)
        canvas[r:r + ph, c:c + pw] = np.maximum(canvas[r:r + ph, c:c + pw], pred[:ph, :pw])

    placed = from_pred + from_ann
    print(f"Placed {placed} patches  "
          f"({from_pred} predicted, {from_ann} from annotations, {missing} missing)")

    profile = {"driver": "GTiff", "dtype": "uint8", "width": img_w,
               "height": img_h, "count": 1, "compress": "lzw"}
    if has_georef and crs is not None:
        profile["crs"]       = crs
        profile["transform"] = transform

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(canvas[np.newaxis, :, :])

    print(f"Stitched -> {rel_to_root(out_path)}")
    print(f"  {pixel_label} pixels: {(canvas > 0).sum():,}  "
          f"({100 * (canvas > 0).mean():.2f}% of image)")

    return out_path, {"transform": transform, "crs": crs, "has_georef": has_georef}
