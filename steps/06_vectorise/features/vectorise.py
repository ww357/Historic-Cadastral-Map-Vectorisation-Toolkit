"""
Vectorise a stitched MapSAM feature mask into simplified polygons and write to GeoPackage.

Any feature label used during annotation can be vectorised — there is no fixed
feature list.  Pass --feature with the same label name used in labelme.

Reads  : data/stitched/<FEATURE>/<SHEET_ID>.tif
Writes : data/outputs/<SHEET_ID>.gpkg  — layer "<feature>"         (polygons)
                                       — layer "<feature>_raster"  (binary prediction)

Pipeline:
  1. Load binary mask from stitched GeoTIFF
  2. Polygonize using rasterio.features.shapes  (equivalent to GDAL Polygonize)
  3. Convert to Shapely geometries + GeoDataFrame
  4. Simplify with Douglas-Peucker  (tolerance from config)
  5. Filter by minimum polygon area  (from config)
  6. Write polygon layer to GeoPackage
  7. Write raster layer to GeoPackage  (for visual comparison)

If the GeoPackage already exists (e.g. from a previous feature run),
the feature layers are replaced without touching other layers.

Per-feature config is read from vectorise.features.<feature> in config.yaml.
If no entry exists for the specific feature, vectorise.features.default is used.

Usage:
    python vectorise.py --sheet Timberscombe --feature water
    python vectorise.py --sheet Timberscombe --feature building
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
from rasterio.features import shapes
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
    """
    Return vectorise config for the given feature.
    Looks for vectorise.features.<feature> first, then vectorise.features.default.
    """
    vcfg = cfg.get("vectorise", {}).get("features", {})
    if feature in vcfg:
        return vcfg[feature]
    if "default" in vcfg:
        return vcfg["default"]
    # Fallback values if nothing is configured
    return {"simplify_tolerance": 2.0, "min_area": 25.0}


def extract_polygons(mask: np.ndarray, transform, has_georef: bool,
                     simplify_tol: float, min_area: float) -> list:
    """
    Polygonize a binary mask and return simplified, filtered Shapely polygons.

    rasterio.features.shapes yields (geojson_geom, value) pairs for each
    connected region.  We keep only the foreground (value > 0) polygons,
    convert to Shapely, simplify, and filter by area.
    """
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
        geom = make_valid(geom)     # repair any self-intersections from raster artefacts

        # Simplify — Douglas-Peucker, same parameter as boundaries
        geom = geom.simplify(simplify_tol, preserve_topology=True)

        if geom.is_empty:
            continue

        # A simplified result may be a MultiPolygon — keep each part individually
        parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for part in parts:
            if not part.is_empty and part.area >= min_area:
                polygons.append(part)

    return polygons


def _layer_exists(gpkg_path: Path, layer_name: str) -> bool:
    """Return True if a layer (vector or raster) already exists in the GeoPackage."""
    if not gpkg_path.exists():
        return False
    con = sqlite3.connect(gpkg_path)
    try:
        cur = con.execute(
            "SELECT 1 FROM gpkg_contents WHERE table_name = ?", (layer_name,)
        )
        return cur.fetchone() is not None
    except sqlite3.OperationalError:
        return False
    finally:
        con.close()


def _write_patch_grid(gpkg_path: Path, meta_path: Path, transform,
                      crs, has_georef: bool, sheet_id: str):
    """
    Write a Patch_Grid polygon layer to the GeoPackage showing the extent of every
    patch as a rectangle.  Skipped if the layer already exists (written by a previous
    feature's vectorise run for the same sheet).

    This lets the user see exactly where patch borders fall on the document and
    identify features that cross patch boundaries.
    """
    if _layer_exists(gpkg_path, "Patch_Grid"):
        return

    if not meta_path.exists():
        print("  Patch_Grid: metadata CSV not found — skipping.")
        return

    meta = pd.read_csv(meta_path)
    rectangles = []

    for _, row in meta.iterrows():
        r  = int(row.row_off)
        c  = int(row.col_off)
        ph = int(row.patch_h)
        pw = int(row.patch_w)

        if has_georef:
            # transform * (col, row) → (x, y) in the raster's CRS
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


def _drop_vector_layer(gpkg_path: Path, layer_name: str):
    """Remove a vector layer from a GeoPackage, leaving all other layers intact."""
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
        pass
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
    src = None


def vectorise(sheet_id: str, feature: str, repo_root: Path):
    cfg  = load_config()
    fcfg = feature_config(cfg, feature)

    simplify_tol = float(fcfg["simplify_tolerance"])
    min_area     = float(fcfg["min_area"])

    stitched_path = repo_root / cfg["paths"]["stitched"] / feature / f"{sheet_id}.tif"
    meta_path     = repo_root / cfg["paths"]["patches"] / "metadata" / f"{sheet_id}_patches.csv"
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

    print(f"Sheet         : {sheet_id}")
    print(f"Feature       : {feature}")
    print(f"Mask          : {mask.shape[1]} × {mask.shape[0]} px  "
          f"|  foreground pixels: {(mask > 0).sum():,}")
    print(f"CRS           : {crs or 'none (pixel coords)'}")
    print(f"Simplify tol  : {simplify_tol}  |  min area: {min_area} map units²")

    # --- Polygonize ---
    print("\nPolygonizing...")
    polygons = extract_polygons(mask, transform, has_georef, simplify_tol, min_area)
    print(f"Polygons after filtering: {len(polygons):,}")

    if not polygons:
        print("Warning: no polygons produced — check mask content and config thresholds.")
        return

    # --- Build GeoDataFrame ---
    gdf = gpd.GeoDataFrame(
        {
            "sheet_id": sheet_id,
            "feature":  feature,
            "area":     [p.area for p in polygons],
        },
        geometry=polygons,
        crs=crs if has_georef else None,
    )

    # --- Write vector layer ---
    # Drop only this feature's layer so other layers in the shared GeoPackage
    # are preserved. mode="a" appends to an existing file; "w" creates a new one.
    _drop_vector_layer(out_path, feature)
    write_mode = "a" if out_path.exists() else "w"
    gdf.to_file(out_path, driver="GPKG", layer=feature, mode=write_mode)
    print(f"\nSaved → {out_path.relative_to(repo_root)}")
    print(f"  {feature} (vector):  {len(gdf):,} polygons  |  "
          f"total area: {gdf['area'].sum():,.1f} map units²")

    # --- Write raster layer ---
    raster_layer = f"{feature}_raster"
    print(f"  Adding {raster_layer}...")
    add_raster_layer(stitched_path, out_path, raster_layer)
    print(f"  {raster_layer} (raster): done")

    # --- Write patch grid (once per sheet — skipped if already present) ---
    _write_patch_grid(out_path, meta_path, transform, crs, has_georef, sheet_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vectorise MapSAM feature mask into polygons and write to GeoPackage."
    )
    parser.add_argument("--sheet",   required=True, help="Sheet ID")
    parser.add_argument("--feature", required=True,
                        help="Feature class — any label used in labelme annotations "
                             "(e.g. water, building, vegetation)")
    args = parser.parse_args()
    vectorise(args.sheet, args.feature, ROOT)
