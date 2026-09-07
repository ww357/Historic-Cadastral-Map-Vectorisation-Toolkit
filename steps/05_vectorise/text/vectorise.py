"""
Write the existing text_preds.geojson into the GeoPackage as a "text" layer.

Run this in the maptools environment after text/predict.py has completed:
    conda activate maptools
    python steps/05_vectorise/text/vectorise.py --sheet SHEET_ID
    python steps/05_vectorise/text/vectorise.py --sheet SHEET_ID --mended

Output target:
  default   data/outputs/<SHEET_ID>.gpkg
  --mended  the hand-corrected GeoPackage in paths.outputs_mended, so the "text"
            layer lands alongside layers already mended in QGIS. Only the "text"
            layer is replaced; the rest are preserved. Errors if none exists.
  --gpkg    an explicit path, overriding both.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import geopandas as gpd
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "steps"))   # shared helpers
from common import load_config, rel_to_root, resolve_output_gpkg   # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Write text_preds.geojson into the sheet GeoPackage as a 'text' layer."
    )
    parser.add_argument("--sheet", required=True)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--mended", action="store_true",
                        help="Write into the hand-corrected GeoPackage in "
                             "paths.outputs_mended instead of paths.outputs. Only the "
                             "'text' layer is replaced; other mended layers are "
                             "preserved. Errors if no mended file exists for the sheet.")
    target.add_argument("--gpkg", default=None,
                        help="Explicit output GeoPackage path (overrides --mended "
                             "and the default).")
    args = parser.parse_args()
    sheet_id = args.sheet

    cfg       = yaml.safe_load((ROOT / "config.yaml").read_text())
    paths     = cfg["paths"]
    geojson   = ROOT / paths["predictions"] / "text" / sheet_id / "text_preds.geojson"
    gpkg_path = resolve_output_gpkg(sheet_id, cfg, args.gpkg, args.mended)

    if not geojson.exists():
        sys.exit(f"GeoJSON not found: {geojson}")

    print(f"Output GPKG: {rel_to_root(gpkg_path)}")

    # Ensure outputs directory exists
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {geojson.name} ...")
    gdf = gpd.read_file(geojson)
    print(f"  {len(gdf):,} features   CRS: {gdf.crs}")

    # Fiona cannot handle pandas StringDtype - cast to plain object
    str_cols = gdf.select_dtypes(include="string").columns.tolist()
    if str_cols:
        print(f"  Casting StringDtype columns: {str_cols}")
        gdf[str_cols] = gdf[str_cols].astype(object)

    # Drop existing text layer from GeoPackage (safe if it doesn't exist)
    if gpkg_path.exists():
        print(f"Dropping existing 'text' layer from {gpkg_path.name} ...")
        con = sqlite3.connect(gpkg_path)
        try:
            con.execute("DELETE FROM gpkg_contents WHERE table_name='text'")
            con.execute("DELETE FROM gpkg_geometry_columns WHERE table_name='text'")
            con.execute("DROP TABLE IF EXISTS [text]")
            con.commit()
        except Exception as e:
            print(f"  (sqlite3 note: {e})")
        finally:
            con.close()
        mode = "a"
    else:
        mode = "w"

    print(f"Writing to {gpkg_path.name}  (mode={mode!r}) ...")
    gdf.to_file(str(gpkg_path), driver="GPKG", layer="text", mode=mode)
    print(f"\nDone - {len(gdf):,} text instances in 'text' layer of {gpkg_path.name}")
    print(
        f"\nNext step: open {gpkg_path.name} in QGIS and review the 'text' layer.\n"
        f"  There is no feedback loop for text - the Rumsey weights are used as-is\n"
        f"  (see CONTEXT.md 'What Was Deliberately NOT Done')."
    )


if __name__ == "__main__":
    main()
