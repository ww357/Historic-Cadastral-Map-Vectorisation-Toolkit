"""
Vectorise a stitched boundary mask into simplified polylines and write to GeoPackage.

Reads  : data/stitched/boundaries/<SHEET_ID>.tif
Writes : data/outputs/<SHEET_ID>.gpkg  — layer "boundaries"      (polylines)
                                       — layer "boundary_raster"  (binary prediction)

Pipeline:
  1. Load binary mask
  2. Skeletonize → 1px centrelines (Lee 1994, topology-preserving)
  3. Extract polylines from skeleton graph via skan
  4. Convert pixel coords → world coords using GeoTIFF transform
  5. Simplify with Douglas-Peucker (tolerance from config)
  6. Filter by minimum line length
  7. Write "boundaries" vector layer to GeoPackage
  8. Write "boundary_raster" raster layer to GeoPackage (for visual comparison)

If the GeoPackage already exists (e.g. from a previous feature run),
both layers are replaced without touching other layers.

Usage:
    python vectorise.py --sheet SHEET_ID
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from osgeo import gdal
from shapely.geometry import LineString
from skimage.morphology import skeletonize
from skan import Skeleton
from tqdm import tqdm

from topology_repair import repair_topology

ROOT = Path(__file__).resolve().parents[3]


def load_config() -> dict:
    p = ROOT / "config.yaml"
    if not p.exists():
        sys.exit(f"config.yaml not found at {p}")
    return yaml.safe_load(p.read_text())


def pixel_to_world(rows, cols, transform) -> list[tuple]:
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    return list(zip(xs, ys))


def extract_polylines(skeleton: np.ndarray, transform, has_georef: bool,
                      simplify_tol: float, min_length: float) -> list[LineString]:
    """
    Walk the skeleton graph with skan and return a list of simplified LineStrings.
    Each skan 'path' is a sequence of connected pixels between two junction/endpoint
    pixels — the natural segments for polyline vectorisation.
    """
    if not skeleton.any():
        return []

    skel_obj = Skeleton(skeleton, keep_images=False)
    lines = []

    for i in tqdm(range(skel_obj.n_paths), desc="Tracing paths", unit="path", leave=False):
        coords = skel_obj.path_coordinates(i)   # (N, 2) — rows, cols
        if len(coords) < 2:
            continue

        rows, cols = coords[:, 0], coords[:, 1]
        pts = pixel_to_world(rows, cols, transform) if has_georef \
              else [(float(c), float(r)) for r, c in zip(rows, cols)]

        line = LineString(pts).simplify(simplify_tol, preserve_topology=True)
        if not line.is_empty and line.length >= min_length:
            lines.append(line)

    return lines


def _drop_raster_layer(gpkg_path: Path, table_name: str):
    """Remove an existing raster tile table from a GeoPackage for clean re-runs."""
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
        pass  # tables may not exist yet on first run
    finally:
        con.close()


def add_raster_layer(stitched_path: Path, gpkg_path: Path, layer_name: str):
    """Write the stitched binary mask as a tiled raster layer in the GeoPackage."""
    _drop_raster_layer(gpkg_path, layer_name)

    src = gdal.Open(str(stitched_path))
    if src is None:
        print(f"Warning: GDAL could not open {stitched_path} — raster layer skipped.")
        return

    gdal.Translate(
        str(gpkg_path),
        src,
        format="GPKG",
        creationOptions=[
            f"RASTER_TABLE={layer_name}",
            "APPEND_SUBDATASET=YES",
        ],
    )
    src = None  # close dataset


def vectorise(sheet_id: str, repo_root: Path):
    cfg  = load_config()
    vcfg = cfg["vectorise"]["boundaries"]

    simplify_tol  = float(vcfg["simplify_tolerance"])
    min_length    = float(vcfg["min_length"])
    repair_cfg    = vcfg.get("topology_repair", {})
    do_repair     = repair_cfg.get("enabled", False)

    stitched_path = repo_root / cfg["paths"]["stitched"] / "boundaries" / f"{sheet_id}.tif"
    out_dir       = repo_root / cfg["paths"]["outputs"]
    out_path      = out_dir / f"{sheet_id}.gpkg"

    if not stitched_path.exists():
        sys.exit(f"Stitched mask not found: {stitched_path}  — run 05_stitch first.")

    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(stitched_path) as src:
        mask       = src.read(1)
        transform  = src.transform
        crs        = src.crs
        has_georef = src.crs is not None

    print(f"Sheet        : {sheet_id}")
    print(f"Mask         : {mask.shape[1]} × {mask.shape[0]} px  "
          f"|  boundary pixels: {(mask > 0).sum():,}")
    print(f"CRS          : {crs or 'none (pixel coords)'}")
    print(f"Simplify tol : {simplify_tol}  |  min length: {min_length}")

    # --- Skeletonize ---
    print("\nSkeletonizing...")
    binary   = mask > 0
    skeleton = skeletonize(binary)
    print(f"Skeleton pixels: {skeleton.sum():,}  (reduced from {binary.sum():,})")

    # --- Extract polylines ---
    print("Extracting polylines...")
    lines = extract_polylines(skeleton, transform, has_georef, simplify_tol, min_length)
    print(f"Polylines after filtering: {len(lines):,}")

    if not lines:
        print("Warning: no polylines produced — check mask content and config thresholds.")
        return

    # --- Build GeoDataFrame ---
    gdf = gpd.GeoDataFrame(
        {"sheet_id": sheet_id, "length": [l.length for l in lines]},
        geometry=lines,
        crs=crs if has_georef else None,
    )

    # --- Optional topology repair ---
    if do_repair:
        snap_dist       = float(repair_cfg.get("snap_distance", 15.0))
        angle_tolerance = repair_cfg.get("angle_tolerance", None)
        if angle_tolerance is not None:
            angle_tolerance = float(angle_tolerance)
        print(f"\nTopology repair  snap={snap_dist} CRS units"
              + (f"  angle≤{angle_tolerance}°" if angle_tolerance else "  no angle filter"))
        gdf = repair_topology(gdf, snap_distance=snap_dist, angle_tolerance=angle_tolerance)
        n_bridges = int(gdf["is_bridge"].sum())
        print(f"  {n_bridges} bridge segment(s) added")

    # --- Write vector layer ---
    gdf.to_file(out_path, driver="GPKG", layer="boundaries", mode="w")
    print(f"\nSaved → {out_path.relative_to(repo_root)}")
    print(f"  boundaries (vector):  {len(gdf):,} features  |  "
          f"total length: {gdf['length'].sum():,.1f} map units"
          + (f"  ({int(gdf['is_bridge'].sum())} bridges)" if do_repair else ""))

    # --- Write raster layer ---
    print("  Adding raster layer...")
    add_raster_layer(stitched_path, out_path, "boundary_raster")
    print("  boundary_raster (raster): done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vectorise boundary mask into polylines and write to GeoPackage."
    )
    parser.add_argument("--sheet", required=True, help="Sheet ID")
    args = parser.parse_args()
    vectorise(args.sheet, ROOT)
