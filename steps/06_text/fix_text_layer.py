"""
One-off recovery script: writes the existing text_preds.geojson into the
GeoPackage without re-running inference.

Run from the project root in the New-MapReader environment:
    python fix_text_layer.py --sheet Timberscombe
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import geopandas as gpd
import yaml

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheet", required=True)
    args = parser.parse_args()
    sheet_id = args.sheet

    cfg      = yaml.safe_load((ROOT / "config.yaml").read_text())
    paths    = cfg["paths"]
    geojson  = ROOT / paths["predictions"] / "text" / sheet_id / "text_preds.geojson"
    gpkg_path = ROOT / paths["outputs"] / f"{sheet_id}.gpkg"

    if not geojson.exists():
        sys.exit(f"GeoJSON not found: {geojson}")

    # Ensure outputs directory exists
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {geojson.name} ...")
    gdf = gpd.read_file(geojson)
    print(f"  {len(gdf):,} features   CRS: {gdf.crs}")

    # Fiona cannot handle pandas StringDtype — cast to plain object
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
    print(f"\nDone — {len(gdf):,} text instances in 'text' layer of {gpkg_path.name}")


if __name__ == "__main__":
    main()
