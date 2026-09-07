"""
Stitch dashed-line predictions into a full-sheet GeoTIFF, then vectorise to GeoPackage.

Same pipeline as steps/05_vectorise/lines/vectorise.py (stitch -> skeletonize ->
skan polyline trace -> Douglas-Peucker -> optional topology repair -> GeoPackage),
writing the "dashed" / "dashed_raster" layers. Stitching, GeoPackage helpers, and
topology repair are all shared modules - see that script for the general method.

Known limitation (see config.yaml `dashed:` and models/DashedLineUNet/): the model
does not discriminate the annotated dashed line from other faint lines sharing a
patch, so expect more false-positive traces than the "boundaries" layer.
dashed.predict_threshold is tuned toward precision to reduce this.

Output target: default data/outputs/<SHEET_ID>.gpkg; --mended writes into the
hand-corrected GeoPackage; --gpkg overrides both.

Usage:
    python vectorise.py --sheet MapSheetName [--mended | --gpkg PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import LineString
from skimage.morphology import skeletonize
from skan import Skeleton
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "steps"))                                 # shared helpers
sys.path.insert(0, str(ROOT / "steps" / "05_vectorise" / "lines"))     # topology_repair

from common import load_config, rel_to_root, resolve_output_gpkg   # noqa: E402
from geopackage import (add_raster_layer, announce_target,          # noqa: E402
                        drop_vector_layer, write_patch_grid)
from vectorise_common import stitch                                 # noqa: E402
from topology_repair import repair_topology                         # noqa: E402

LAYER = "dashed"


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


def vectorise(sheet_id: str, cfg: dict, stitched_path: Path, georef: dict,
              out_path: Path) -> None:
    vcfg         = cfg["vectorise"][LAYER]
    simplify_tol = float(vcfg["simplify_tolerance"])
    min_length   = float(vcfg["min_length"])
    repair_cfg   = vcfg.get("topology_repair", {})
    do_repair    = repair_cfg.get("enabled", False)

    meta_path = ROOT / cfg["paths"]["patches"] / "metadata" / f"{sheet_id}_patches.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    transform, crs, has_georef = georef["transform"], georef["crs"], georef["has_georef"]

    with rasterio.open(stitched_path) as src:
        mask = src.read(1)

    print(f"\n-- Vectorise -----------------------------------")
    print(f"Mask         : {mask.shape[1]} x {mask.shape[0]} px  "
          f"|  {LAYER} pixels: {(mask > 0).sum():,}")
    print(f"CRS          : {crs or 'none (pixel coords)'}")
    print(f"Simplify tol : {simplify_tol}  |  min length: {min_length}")
    announce_target(out_path, [LAYER, f"{LAYER}_raster", "Patch_Grid"])

    print("\nSkeletonizing...")
    binary   = mask > 0
    skeleton = skeletonize(binary)
    print(f"Skeleton pixels: {skeleton.sum():,}  (reduced from {binary.sum():,})")

    print("Extracting polylines...")
    lines = extract_polylines(skeleton, transform, has_georef, simplify_tol, min_length)
    print(f"Polylines after filtering: {len(lines):,}")

    if not lines:
        print("Warning: no polylines produced - check mask and config thresholds.")
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

    drop_vector_layer(out_path, LAYER)
    write_mode = "a" if out_path.exists() else "w"
    gdf.to_file(out_path, driver="GPKG", layer=LAYER, mode=write_mode)
    print(f"\nSaved -> {rel_to_root(out_path)}")
    print(f"  {LAYER} (vector):  {len(gdf):,} features  |  "
          f"total length: {gdf['length'].sum():,.1f} map units"
          + (f"  ({int(gdf['is_bridge'].sum())} bridges)" if do_repair else ""))

    print("  Adding raster layer...")
    add_raster_layer(stitched_path, out_path, f"{LAYER}_raster")
    print(f"  {LAYER}_raster (raster): done")

    write_patch_grid(out_path, meta_path, transform, crs, has_georef, sheet_id, cfg)

    # No feedback loop for dashed yet (the model is pooled across sheets, not
    # per-sheet fine-tuned) - the useful onward step is parcel extraction, which
    # can use these dashed lines as an extra wall.
    print(
        f"\nNext step: mend the '{LAYER}' layer in QGIS. To use it as a parcel boundary:\n"
        f"  conda activate polygons\n"
        f"  python steps/04_predict/parcels/predict.py --sheet {sheet_id} --extent boundaries dashed"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stitch dashed-line predictions and vectorise to GeoPackage."
    )
    parser.add_argument("--sheet", required=True, help="Sheet ID")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--mended", action="store_true",
                        help="Write into the hand-corrected GeoPackage in "
                             "paths.outputs_mended instead of paths.outputs.")
    target.add_argument("--gpkg", default=None,
                        help="Explicit output GeoPackage path (overrides --mended and the default).")
    args = parser.parse_args()

    cfg = load_config()
    out_path = resolve_output_gpkg(args.sheet, cfg, args.gpkg, args.mended)
    stitched_path, georef = stitch(args.sheet, cfg, LAYER, LAYER, LAYER, LAYER)
    vectorise(args.sheet, cfg, stitched_path, georef, out_path)


if __name__ == "__main__":
    main()
