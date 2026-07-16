"""
Stitch dashed-line predictions into a full-sheet GeoTIFF, then vectorise to GeoPackage.

Reads  : data/patches/metadata/<SHEET_ID>_patches.csv     — patch offsets + georef
         data/predictions/dashed/<SHEET_ID>/*.png         — 512px binary masks
         data/annotations/dashed/<SHEET_ID>/masks/        — annotation masks (fallback,
                                                             includes confirmed-negative patches)
         data/raw/<SHEET_ID>/<SHEET_ID>.tif                — source dimensions + CRS

Writes : data/stitched/dashed/<SHEET_ID>.tif              — full-sheet uint8 GeoTIFF
         data/outputs/<SHEET_ID>.gpkg                     — layer "dashed" (polylines)
                                                            — layer "dashed_raster" (raster)
                                                            — layer "Patch_Grid" (rebuilt)

Same pipeline as steps/05_vectorise/lines/vectorise.py (stitch -> skeletonize ->
skan polyline trace -> Douglas-Peucker -> optional topology repair -> GeoPackage),
reusing that module's topology_repair implementation directly rather than
duplicating it. See that script's docstring for the general method.

Known limitation carried over from the model (see config.yaml `dashed:`
section and models/DashedLineUNet/architecture.py): the model does not yet
discriminate the annotated dashed line from other faint boundary lines
sharing a patch, so expect more false-positive traces — and more mending
time — than the "boundaries" layer, especially in visually cluttered
patches. dashed.predict_threshold is already tuned toward precision (best-F1
from a threshold sweep) to reduce this, but it will not eliminate it.

Usage:
    conda activate maptools
    python steps/05_vectorise/dashed/vectorise.py --sheet SHEET_ID
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
sys.path.insert(0, str(ROOT / "steps" / "05_vectorise" / "lines"))  # reuse topology_repair

from topology_repair import repair_topology  # noqa: E402

LAYER = "dashed"


def load_config() -> dict:
    p = ROOT / "config.yaml"
    if not p.exists():
        sys.exit(f"config.yaml not found at {p}")
    return yaml.safe_load(p.read_text())


# ---------------------------------------------------------------------------
# Stitch
# ---------------------------------------------------------------------------

def stitch(sheet_id: str, cfg: dict) -> tuple[Path, dict]:
    paths = cfg["paths"]

    raw_path     = ROOT / paths["raw"]         / sheet_id / f"{sheet_id}.tif"
    meta_path    = ROOT / paths["patches"]     / "metadata" / f"{sheet_id}_patches.csv"
    pred_dir     = ROOT / paths["predictions"] / LAYER / sheet_id
    ann_mask_dir = ROOT / paths["annotations"] / LAYER / sheet_id / "masks"
    out_dir      = ROOT / paths["stitched"]    / LAYER
    out_path     = out_dir / f"{sheet_id}.tif"

    if not meta_path.exists():
        sys.exit(f"Metadata CSV not found: {meta_path}")
    if not pred_dir.exists():
        print(f"Warning: predictions dir not found: {pred_dir}")
        print("  Annotation masks will be used where available; other patches will be blank.")

    out_dir.mkdir(parents=True, exist_ok=True)

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
        r, c   = int(row.row_off), int(row.col_off)
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

    print(f"Stitched -> {out_path.relative_to(ROOT)}")
    print(f"  Dashed-line pixels: {(canvas > 0).sum():,}  "
          f"({100 * (canvas > 0).mean():.2f}% of image)")

    return out_path, {"transform": transform, "crs": crs, "has_georef": has_georef}


# ---------------------------------------------------------------------------
# GeoPackage helpers (duplicated from lines/vectorise.py by convention —
# each vectorise script owns its own copy; see that script for rationale)
# ---------------------------------------------------------------------------

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
                      crs, has_georef: bool, sheet_id: str, cfg: dict):
    """Rebuild the Patch_Grid layer every run so annotation columns stay current."""
    _drop_vector_layer(gpkg_path, "Patch_Grid")
    if not meta_path.exists():
        print("  Patch_Grid: metadata CSV not found — skipping.")
        return

    ann_root = ROOT / cfg["paths"]["annotations"]
    feature_mask_dirs: dict[str, Path] = {}
    if ann_root.exists():
        for feat_dir in sorted(ann_root.iterdir()):
            if not feat_dir.is_dir():
                continue
            mask_dir = feat_dir / sheet_id / "masks"
            if mask_dir.exists():
                feature_mask_dirs[feat_dir.name] = mask_dir

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

        rec = {"patch_id": row.patch_id, "sheet_id": sheet_id, "geometry": geom}

        annotated = []
        for feature, mask_dir in feature_mask_dirs.items():
            has_ann = (mask_dir / f"{row.patch_id}.png").exists()
            rec[f"ann_{feature}"] = has_ann
            if has_ann:
                annotated.append(feature)

        rec["annotated_features"] = ", ".join(annotated)
        rectangles.append(rec)

    grid_gdf = gpd.GeoDataFrame(rectangles, crs=crs if has_georef else None)
    write_mode = "a" if gpkg_path.exists() else "w"
    grid_gdf.to_file(gpkg_path, driver="GPKG", layer="Patch_Grid", mode=write_mode)
    ann_cols = [f"ann_{f}" for f in feature_mask_dirs] or ["(none)"]
    print(f"  Patch_Grid (vector):  {len(grid_gdf):,} patches  |  "
          f"annotation columns: {', '.join(ann_cols)}")


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
    vcfg         = cfg["vectorise"][LAYER]
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

    print(f"\n-- Vectorise -------------------------------------")
    print(f"Mask         : {mask.shape[1]} x {mask.shape[0]} px  "
          f"|  dashed-line pixels: {(mask > 0).sum():,}")
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
              + (f"  angle<={angle_tolerance} deg" if angle_tolerance else "  no angle filter"))
        gdf       = repair_topology(gdf, snap_distance=snap_dist, angle_tolerance=angle_tolerance)
        n_bridges = int(gdf["is_bridge"].sum())
        print(f"  {n_bridges} bridge segment(s) added")

    _drop_vector_layer(out_path, LAYER)
    write_mode = "a" if out_path.exists() else "w"
    gdf.to_file(out_path, driver="GPKG", layer=LAYER, mode=write_mode)
    print(f"\nSaved -> {out_path.relative_to(ROOT)}")
    print(f"  {LAYER} (vector):  {len(gdf):,} features  |  "
          f"total length: {gdf['length'].sum():,.1f} map units"
          + (f"  ({int(gdf['is_bridge'].sum())} bridges)" if do_repair else ""))

    print("  Adding raster layer...")
    _add_raster_layer(stitched_path, out_path, f"{LAYER}_raster")
    print(f"  {LAYER}_raster (raster): done")

    _write_patch_grid(out_path, meta_path, transform, crs, has_georef, sheet_id, cfg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stitch dashed-line predictions and vectorise to GeoPackage."
    )
    parser.add_argument("--sheet", required=True, help="Sheet ID")
    args = parser.parse_args()

    cfg = load_config()
    stitched_path, georef = stitch(args.sheet, cfg)
    vectorise(args.sheet, cfg, stitched_path, georef)


if __name__ == "__main__":
    main()
