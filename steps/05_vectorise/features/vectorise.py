"""
Stitch MapSAM feature predictions into a full-sheet GeoTIFF, then vectorise to GeoPackage.

Any feature label used during annotation can be processed — there is no fixed list.
Pass --feature with the same label name used in labelme.

Reads  : data/patches/metadata/<SHEET_ID>_patches.csv          — patch offsets + georef
         data/predictions/<FEATURE>/<SHEET_ID>/*.png           — 512px binary masks
         data/annotations/<FEATURE>/<SHEET_ID>/masks/*.png     — annotation masks (fallback)
         data/raw/<SHEET_ID>/<SHEET_ID>.tif                    — source dimensions + CRS

Writes : data/stitched/<FEATURE>/<SHEET_ID>.tif                — full-sheet uint8 GeoTIFF
         data/outputs/<SHEET_ID>.gpkg                          — layer "<feature>" (polygons)
                                                               — layer "<feature>_raster" (raster)
                                                               — layer "Patch_Grid" (once per sheet)

Pipeline:
  1. Reassemble 512px prediction patches onto a full-sheet canvas
     (annotated patches use their annotation mask directly — skips model for those areas)
  2. Save stitched GeoTIFF
  3. Polygonize using rasterio.features.shapes (equivalent to GDAL Polygonize)
  4. Simplify with Douglas-Peucker + filter by minimum area
  5. Write to GeoPackage alongside raster and patch grid layers

Per-feature config is read from vectorise.features.<feature> in config.yaml.
Falls back to vectorise.features.default if no specific entry exists.

Usage:
    python vectorise.py --sheet MapSheetName --feature FeatureName
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from osgeo import gdal
from PIL import Image
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import box, shape
from shapely.validation import make_valid
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]


def load_config() -> dict:
    p = ROOT / "config.yaml"
    if not p.exists():
        sys.exit(f"config.yaml not found at {p}")
    return yaml.safe_load(p.read_text())


def feature_config(cfg: dict, feature: str) -> dict:
    vcfg = cfg.get("vectorise", {}).get("features", {})
    if feature in vcfg:
        return vcfg[feature]
    if "default" in vcfg:
        return vcfg["default"]
    return {"simplify_tolerance": 2.0, "min_area": 25.0}


# ---------------------------------------------------------------------------
# Stitch
# ---------------------------------------------------------------------------

def stitch(sheet_id: str, feature: str, cfg: dict) -> tuple[Path, dict]:
    """
    Reassemble patch predictions (+ annotation fallbacks) into a full-sheet
    GeoTIFF.  Returns (stitched_path, georef_dict) for the vectorise stage.
    """
    paths = cfg["paths"]

    raw_path     = ROOT / paths["raw"]         / sheet_id / f"{sheet_id}.tif"
    meta_path    = ROOT / paths["patches"]     / "metadata" / f"{sheet_id}_patches.csv"
    pred_dir     = ROOT / paths["predictions"] / feature / sheet_id
    ann_mask_dir = ROOT / paths["annotations"] / feature / sheet_id / "masks"
    out_dir      = ROOT / paths["stitched"]    / feature
    out_path     = out_dir / f"{sheet_id}.tif"

    if not meta_path.exists():
        sys.exit(f"Metadata CSV not found: {meta_path}")
    if not pred_dir.exists():
        print(f"Warning: predictions dir not found: {pred_dir}")
        print("  Annotation masks will be used where available; other patches will be blank.")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Source dimensions + georef
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

    print(f"\n── Stitch ──────────────────────────────────────")
    print(f"Sheet      : {sheet_id}  |  Feature: {feature}")
    print(f"Canvas     : {img_w} × {img_h} px  |  CRS: {crs or 'none'}")

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

        ph = int(row.patch_h)
        pw = int(row.patch_w)
        r  = int(row.row_off)
        c  = int(row.col_off)
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

    n_px = int((canvas > 0).sum())
    print(f"Stitched → {out_path.relative_to(ROOT)}")
    print(f"  {feature} pixels: {n_px:,}  ({100 * n_px / (img_w * img_h):.2f}% of image)")

    return out_path, {"transform": transform, "crs": crs, "has_georef": has_georef}


# ---------------------------------------------------------------------------
# GeoPackage helpers
# ---------------------------------------------------------------------------

def _layer_exists(gpkg_path: Path, layer_name: str) -> bool:
    if not gpkg_path.exists():
        return False
    con = sqlite3.connect(gpkg_path)
    try:
        cur = con.execute("SELECT 1 FROM gpkg_contents WHERE table_name = ?", (layer_name,))
        return cur.fetchone() is not None
    except sqlite3.OperationalError:
        return False
    finally:
        con.close()


def _drop_vector_layer(gpkg_path: Path, layer_name: str):
    if not gpkg_path.exists():
        return
    con = sqlite3.connect(gpkg_path)
    try:
        con.execute("DELETE FROM gpkg_contents WHERE table_name = ?",         (layer_name,))
        con.execute("DELETE FROM gpkg_geometry_columns WHERE table_name = ?", (layer_name,))
        con.execute(f"DROP TABLE IF EXISTS [{layer_name}]")
        con.commit()
    except sqlite3.OperationalError:
        pass
    finally:
        con.close()


def _drop_raster_layer(gpkg_path: Path, table_name: str):
    if not gpkg_path.exists():
        return
    con = sqlite3.connect(gpkg_path)
    try:
        con.execute("DELETE FROM gpkg_contents WHERE table_name = ?",        (table_name,))
        con.execute("DELETE FROM gpkg_tile_matrix_set WHERE table_name = ?", (table_name,))
        con.execute("DELETE FROM gpkg_tile_matrix WHERE table_name = ?",     (table_name,))
        con.execute(f"DROP TABLE IF EXISTS [{table_name}]")
        con.commit()
    except sqlite3.OperationalError:
        pass
    finally:
        con.close()


def _add_raster_layer(stitched_path: Path, gpkg_path: Path, layer_name: str):
    _drop_raster_layer(gpkg_path, layer_name)
    src = gdal.Open(str(stitched_path))
    if src is None:
        print(f"Warning: GDAL could not open {stitched_path} — raster layer skipped.")
        return
    gdal.Translate(str(gpkg_path), src, format="GPKG",
                   creationOptions=[f"RASTER_TABLE={layer_name}", "APPEND_SUBDATASET=YES"])
    src = None


def _write_patch_grid(gpkg_path: Path, meta_path: Path, transform,
                      crs, has_georef: bool, sheet_id: str):
    """Write a Patch_Grid layer once per sheet — skipped if already present."""
    if _layer_exists(gpkg_path, "Patch_Grid"):
        return
    if not meta_path.exists():
        print("  Patch_Grid: metadata CSV not found — skipping.")
        return
    meta = pd.read_csv(meta_path)
    rectangles = []
    for _, row in meta.iterrows():
        r, c, ph, pw = int(row.row_off), int(row.col_off), int(row.patch_h), int(row.patch_w)
        if has_georef:
            x0, y0 = transform * (c,      r)
            x1, y1 = transform * (c + pw, r + ph)
            geom = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        else:
            geom = box(c, r, c + pw, r + ph)
        rectangles.append({"patch_id": row.patch_id, "sheet_id": sheet_id, "geometry": geom})
    grid_gdf = gpd.GeoDataFrame(rectangles, crs=crs if has_georef else None)
    write_mode = "a" if gpkg_path.exists() else "w"
    grid_gdf.to_file(gpkg_path, driver="GPKG", layer="Patch_Grid", mode=write_mode)
    print(f"  Patch_Grid (vector):  {len(grid_gdf):,} patches")


# ---------------------------------------------------------------------------
# Vectorise
# ---------------------------------------------------------------------------

def extract_polygons(mask: np.ndarray, transform, has_georef: bool,
                     simplify_tol: float, min_area: float) -> list:
    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        return []
    polygons = []
    gen = shapes(binary, mask=binary, connectivity=8,
                 transform=transform if has_georef else rasterio.transform.IDENTITY)
    for geom_dict, value in tqdm(gen, desc="Polygonizing", unit="region", leave=False):
        if value == 0:
            continue
        geom = shape(geom_dict)
        geom = make_valid(geom)
        geom = geom.simplify(simplify_tol, preserve_topology=True)
        if geom.is_empty:
            continue
        parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for part in parts:
            if not part.is_empty and part.area >= min_area:
                polygons.append(part)
    return polygons


def vectorise(sheet_id: str, feature: str, cfg: dict, stitched_path: Path, georef: dict):
    fcfg         = feature_config(cfg, feature)
    simplify_tol = float(fcfg["simplify_tolerance"])
    min_area     = float(fcfg["min_area"])

    meta_path = ROOT / cfg["paths"]["patches"] / "metadata" / f"{sheet_id}_patches.csv"
    out_dir   = ROOT / cfg["paths"]["outputs"]
    out_path  = out_dir / f"{sheet_id}.gpkg"
    out_dir.mkdir(parents=True, exist_ok=True)

    transform  = georef["transform"]
    crs        = georef["crs"]
    has_georef = georef["has_georef"]

    with rasterio.open(stitched_path) as src:
        mask = src.read(1)

    print(f"\n── Vectorise ───────────────────────────────────")
    print(f"Mask          : {mask.shape[1]} × {mask.shape[0]} px  "
          f"|  foreground pixels: {(mask > 0).sum():,}")
    print(f"CRS           : {crs or 'none (pixel coords)'}")
    print(f"Simplify tol  : {simplify_tol}  |  min area: {min_area} map units²")

    print("\nPolygonizing...")
    polygons = extract_polygons(mask, transform, has_georef, simplify_tol, min_area)
    print(f"Polygons after filtering: {len(polygons):,}")

    if not polygons:
        print("Warning: no polygons produced — check mask and config thresholds.")
        return

    gdf = gpd.GeoDataFrame(
        {"sheet_id": sheet_id, "feature": feature, "area": [p.area for p in polygons]},
        geometry=polygons,
        crs=crs if has_georef else None,
    )

    _drop_vector_layer(out_path, feature)
    write_mode = "a" if out_path.exists() else "w"
    gdf.to_file(out_path, driver="GPKG", layer=feature, mode=write_mode)
    print(f"\nSaved → {out_path.relative_to(ROOT)}")
    print(f"  {feature} (vector):  {len(gdf):,} polygons  |  "
          f"total area: {gdf['area'].sum():,.1f} map units²")

    raster_layer = f"{feature}_raster"
    print(f"  Adding {raster_layer}...")
    _add_raster_layer(stitched_path, out_path, raster_layer)
    print(f"  {raster_layer} (raster): done")

    _write_patch_grid(out_path, meta_path, transform, crs, has_georef, sheet_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stitch MapSAM feature predictions and vectorise to GeoPackage."
    )
    parser.add_argument("--sheet",   required=True, help="Sheet ID")
    parser.add_argument("--feature", required=True,
                        help="Feature class — any label used in labelme annotations "
                             "(e.g. water, building, vegetation)")
    args = parser.parse_args()

    cfg = load_config()
    stitched_path, georef = stitch(args.sheet, args.feature, cfg)
    vectorise(args.sheet, args.feature, cfg, stitched_path, georef)


if __name__ == "__main__":
    main()
