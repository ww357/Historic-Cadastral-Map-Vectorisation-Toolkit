"""
Stitch boundary predictions into a full-sheet GeoTIFF, then vectorise to GeoPackage.

Reads  : data/patches/metadata/<SHEET_ID>_patches.csv          — patch offsets + georef
         data/predictions/boundaries/<SHEET_ID>/*.png          — 512px binary masks
         data/annotations/<boundary_label>/<SHEET_ID>/masks/   — annotation masks (fallback)
         data/raw/<SHEET_ID>/<SHEET_ID>.tif                    — source dimensions + CRS

Writes : data/stitched/boundaries/<SHEET_ID>.tif               — full-sheet uint8 GeoTIFF
         data/outputs/<SHEET_ID>.gpkg                          — layer "boundaries" (polylines)
                                                               — layer "boundary_raster" (raster)
                                                               — layer "Patch_Grid" (once per sheet)

Pipeline:
  1. Reassemble 512px prediction patches onto a full-sheet canvas
     (annotated patches use their annotation mask directly — skips model for those areas)
  2. Save stitched GeoTIFF
  3. Skeletonize → 1px centrelines (Lee 1994, topology-preserving)
  4. Extract polylines from skeleton graph via skan
  5. Simplify with Douglas-Peucker + filter by minimum length
  6. Optional topology repair (T-junction snapping + endpoint bridging)
  7. Write to GeoPackage alongside raster and patch grid layers

Usage:
    python vectorise.py --sheet MapSheetName
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
from rasterio.transform import Affine
from shapely.geometry import LineString, box
from skimage.morphology import skeletonize
from skan import Skeleton
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))  # for topology_repair local import

from topology_repair import repair_topology  # noqa: E402


def load_config() -> dict:
    p = ROOT / "config.yaml"
    if not p.exists():
        sys.exit(f"config.yaml not found at {p}")
    return yaml.safe_load(p.read_text())


# ---------------------------------------------------------------------------
# Stitch
# ---------------------------------------------------------------------------

def stitch(sheet_id: str, cfg: dict) -> tuple[Path, dict]:
    """
    Reassemble patch predictions (+ annotation fallbacks) into a full-sheet
    GeoTIFF.  Returns (stitched_path, georef_dict) for the vectorise stage.
    """
    paths          = cfg["paths"]
    boundary_label = cfg["annotation"].get("boundary_label", "boundary")

    raw_path     = ROOT / paths["raw"]         / sheet_id / f"{sheet_id}.tif"
    meta_path    = ROOT / paths["patches"]     / "metadata" / f"{sheet_id}_patches.csv"
    pred_dir     = ROOT / paths["predictions"] / "boundaries" / sheet_id
    ann_mask_dir = ROOT / paths["annotations"] / boundary_label / sheet_id / "masks"
    out_dir      = ROOT / paths["stitched"]    / "boundaries"
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
    print(f"Sheet      : {sheet_id}")
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

    print(f"Stitched → {out_path.relative_to(ROOT)}")
    print(f"  Boundary pixels: {(canvas > 0).sum():,}  "
          f"({100 * (canvas > 0).mean():.2f}% of image)")

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

def pixel_to_world(rows, cols, transform) -> list[tuple]:
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    return list(zip(xs, ys))


def extract_polylines(skeleton: np.ndarray, transform, has_georef: bool,
                      simplify_tol: float, min_length: float) -> list[LineString]:
    if not skeleton.any():
        return []
    skel_obj = Skeleton(skeleton, keep_images=False)
    lines = []
    for i in tqdm(range(skel_obj.n_paths), desc="Tracing paths", unit="path", leave=False):
        coords = skel_obj.path_coordinates(i)
        if len(coords) < 2:
            continue
        rows, cols = coords[:, 0], coords[:, 1]
        pts  = pixel_to_world(rows, cols, transform) if has_georef \
               else [(float(c), float(r)) for r, c in zip(rows, cols)]
        line = LineString(pts).simplify(simplify_tol, preserve_topology=True)
        if not line.is_empty and line.length >= min_length:
            lines.append(line)
    return lines


def vectorise(sheet_id: str, cfg: dict, stitched_path: Path, georef: dict):
    vcfg         = cfg["vectorise"]["boundaries"]
    simplify_tol = float(vcfg["simplify_tolerance"])
    min_length   = float(vcfg["min_length"])
    repair_cfg   = vcfg.get("topology_repair", {})
    do_repair    = repair_cfg.get("enabled", False)

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
    print(f"Mask         : {mask.shape[1]} × {mask.shape[0]} px  "
          f"|  boundary pixels: {(mask > 0).sum():,}")
    print(f"CRS          : {crs or 'none (pixel coords)'}")
    print(f"Simplify tol : {simplify_tol}  |  min length: {min_length}")

    print("\nSkeletonizing...")
    binary   = mask > 0
    skeleton = skeletonize(binary)
    print(f"Skeleton pixels: {skeleton.sum():,}  (reduced from {binary.sum():,})")

    print("Extracting polylines...")
    lines = extract_polylines(skeleton, transform, has_georef, simplify_tol, min_length)
    print(f"Polylines after filtering: {len(lines):,}")

    if not lines:
        print("Warning: no polylines produced — check mask and config thresholds.")
        return

    gdf = gpd.GeoDataFrame(
        {"sheet_id": sheet_id, "length": [l.length for l in lines]},
        geometry=lines,
        crs=crs if has_georef else None,
    )

    if do_repair:
        snap_dist       = float(repair_cfg.get("snap_distance", 15.0))
        angle_tolerance = repair_cfg.get("angle_tolerance", None)
        if angle_tolerance is not None:
            angle_tolerance = float(angle_tolerance)
        print(f"\nTopology repair  snap={snap_dist} CRS units"
              + (f"  angle≤{angle_tolerance}°" if angle_tolerance else "  no angle filter"))
        gdf        = repair_topology(gdf, snap_distance=snap_dist, angle_tolerance=angle_tolerance)
        n_bridges  = int(gdf["is_bridge"].sum())
        print(f"  {n_bridges} bridge segment(s) added")

    _drop_vector_layer(out_path, "boundaries")
    write_mode = "a" if out_path.exists() else "w"
    gdf.to_file(out_path, driver="GPKG", layer="boundaries", mode=write_mode)
    print(f"\nSaved → {out_path.relative_to(ROOT)}")
    print(f"  boundaries (vector):  {len(gdf):,} features  |  "
          f"total length: {gdf['length'].sum():,.1f} map units"
          + (f"  ({int(gdf['is_bridge'].sum())} bridges)" if do_repair else ""))

    print("  Adding raster layer...")
    _add_raster_layer(stitched_path, out_path, "boundary_raster")
    print("  boundary_raster (raster): done")

    _write_patch_grid(out_path, meta_path, transform, crs, has_georef, sheet_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stitch boundary predictions and vectorise to GeoPackage."
    )
    parser.add_argument("--sheet", required=True, help="Sheet ID")
    args = parser.parse_args()

    cfg = load_config()
    stitched_path, georef = stitch(args.sheet, cfg)
    vectorise(args.sheet, cfg, stitched_path, georef)


if __name__ == "__main__":
    main()
