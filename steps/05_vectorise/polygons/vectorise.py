"""
Stitch MapSAM feature predictions into a full-sheet GeoTIFF, then vectorise to GeoPackage.

Any feature label used during annotation can be processed - there is no fixed list.
Pass --feature with the same label name used in labelme; omit it to auto-discover
every MapSAM-predicted feature for the sheet.

Reads  : data/patches/metadata/<SHEET_ID>_patches.csv          - patch offsets + georef
         data/predictions/<FEATURE>/<SHEET_ID>/*.png           - 512px binary masks
         data/annotations/<FEATURE>/<SHEET_ID>/masks/*.png     - annotation masks (fallback)
         data/raw/<SHEET_ID>/<SHEET_ID>.<ext>                  - source dimensions + CRS

Writes : data/stitched/<FEATURE>/<SHEET_ID>.tif                - full-sheet uint8 GeoTIFF
         data/outputs/<SHEET_ID>.gpkg                          - layer "<feature>" (polygons)
                                                               - layer "<feature>_raster" (raster)
                                                               - layer "Patch_Grid" (rebuilt each run)

Per-feature config is read from vectorise.features.<feature> in config.yaml, falling
back to vectorise.features.default. Stitching and the GeoPackage helpers are shared
(steps/vectorise_common.py, steps/geopackage.py).

Output target: default data/outputs; --mended writes into the hand-corrected
GeoPackage; --gpkg overrides both.

Usage:
    python vectorise.py --sheet MapSheetName --feature water
    python vectorise.py --sheet MapSheetName --feature water building vegetation
    python vectorise.py --sheet MapSheetName                  (auto-discovers predicted features)
    python vectorise.py --sheet MapSheetName --mended
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.validation import make_valid
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "steps"))

from common import load_config, rel_to_root, resolve_output_gpkg   # noqa: E402
from geopackage import (add_raster_layer, announce_target,          # noqa: E402
                        drop_vector_layer, write_patch_grid)
from vectorise_common import stitch                                 # noqa: E402

# Feature folders under data/predictions/ that are not MapSAM polygon classes.
_NON_MAPSAM = {"boundaries", "dashed", "text", "parcels"}


def feature_config(cfg: dict, feature: str) -> dict:
    vcfg = cfg.get("vectorise", {}).get("features", {})
    if feature in vcfg:
        return vcfg[feature]
    if "default" in vcfg:
        return vcfg["default"]
    return {"simplify_tolerance": 2.0, "min_area": 25.0}


def discover_features(sheet_id: str, cfg: dict) -> list[str]:
    """All MapSAM-predicted features for this sheet - every data/predictions/<f>/<sheet>/
    folder holding PNGs, minus the non-MapSAM layers."""
    pred_root = ROOT / cfg["paths"]["predictions"]
    if not pred_root.exists():
        return []
    features = []
    for feature_dir in sorted(pred_root.iterdir()):
        if not feature_dir.is_dir() or feature_dir.name in _NON_MAPSAM:
            continue
        sheet_pred_dir = feature_dir / sheet_id
        if sheet_pred_dir.exists() and any(sheet_pred_dir.glob("*.png")):
            features.append(feature_dir.name)
    return features


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
        geom = make_valid(shape(geom_dict)).simplify(simplify_tol, preserve_topology=True)
        if geom.is_empty:
            continue
        parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for part in parts:
            if not part.is_empty and part.area >= min_area:
                polygons.append(part)
    return polygons


def vectorise(sheet_id: str, feature: str, cfg: dict, stitched_path: Path,
              georef: dict, out_path: Path) -> None:
    fcfg         = feature_config(cfg, feature)
    simplify_tol = float(fcfg["simplify_tolerance"])
    min_area     = float(fcfg["min_area"])

    meta_path = ROOT / cfg["paths"]["patches"] / "metadata" / f"{sheet_id}_patches.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    transform, crs, has_georef = georef["transform"], georef["crs"], georef["has_georef"]

    with rasterio.open(stitched_path) as src:
        mask = src.read(1)

    print(f"\n-- Vectorise -----------------------------------")
    print(f"Mask          : {mask.shape[1]} x {mask.shape[0]} px  "
          f"|  foreground pixels: {(mask > 0).sum():,}")
    print(f"CRS           : {crs or 'none (pixel coords)'}")
    print(f"Simplify tol  : {simplify_tol}  |  min area: {min_area} map units2")
    announce_target(out_path, [feature, f"{feature}_raster", "Patch_Grid"])

    print("\nPolygonizing...")
    polygons = extract_polygons(mask, transform, has_georef, simplify_tol, min_area)
    print(f"Polygons after filtering: {len(polygons):,}")

    if not polygons:
        print("Warning: no polygons produced - check mask and config thresholds.")
        return

    gdf = gpd.GeoDataFrame(
        {"sheet_id": sheet_id, "feature": feature, "area": [p.area for p in polygons]},
        geometry=polygons,
        crs=crs if has_georef else None,
    )

    drop_vector_layer(out_path, feature)
    write_mode = "a" if out_path.exists() else "w"
    gdf.to_file(out_path, driver="GPKG", layer=feature, mode=write_mode)
    print(f"\nSaved -> {rel_to_root(out_path)}")
    print(f"  {feature} (vector):  {len(gdf):,} polygons  |  "
          f"total area: {gdf['area'].sum():,.1f} map units2")

    raster_layer = f"{feature}_raster"
    print(f"  Adding {raster_layer}...")
    add_raster_layer(stitched_path, out_path, raster_layer)
    print(f"  {raster_layer} (raster): done")

    write_patch_grid(out_path, meta_path, transform, crs, has_georef, sheet_id, cfg)

    print(
        f"\nNext step: mend the '{feature}' layer in QGIS, then feed the corrections back:\n"
        f"  conda activate polygons\n"
        f"  python steps/06_feedback/polygons/prepare.py --sheet {sheet_id} --feature {feature}\n"
        f"  python steps/06_feedback/polygons/train.py --sheet {sheet_id} --feature {feature}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stitch MapSAM feature predictions and vectorise to GeoPackage."
    )
    parser.add_argument("--sheet",   required=True, help="Sheet ID")
    parser.add_argument("--feature", nargs="+", default=None,
                        help="Feature class(es) to vectorise (e.g. water building). "
                             "If omitted, all MapSAM-predicted features are auto-discovered.")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--mended", action="store_true",
                        help="Write into the hand-corrected GeoPackage in "
                             "paths.outputs_mended instead of paths.outputs.")
    target.add_argument("--gpkg", default=None,
                        help="Explicit output GeoPackage path (overrides --mended and the default).")
    args = parser.parse_args()

    cfg = load_config()
    out_path = resolve_output_gpkg(args.sheet, cfg, args.gpkg, args.mended)

    features = args.feature
    if not features:
        features = discover_features(args.sheet, cfg)
        if not features:
            sys.exit(
                f"No MapSAM predictions found for sheet '{args.sheet}' "
                f"under {ROOT / cfg['paths']['predictions']}.\n"
                "Run 04_predict first, or pass --feature explicitly."
            )
        print(f"Auto-discovered features: {', '.join(features)}\n")

    for i, feature in enumerate(features):
        if len(features) > 1:
            print(f"=== Feature {i + 1}/{len(features)}: {feature} ===\n")
        stitched_path, georef = stitch(args.sheet, cfg, feature, feature, feature, feature)
        vectorise(args.sheet, feature, cfg, stitched_path, georef, out_path)
        if len(features) > 1 and i < len(features) - 1:
            print()


if __name__ == "__main__":
    main()
