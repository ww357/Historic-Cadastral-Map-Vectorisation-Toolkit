"""
GeoPackage layer helpers shared by the vectorise steps.

Each vectorise step writes its own layers into a shared per-sheet GeoPackage
without disturbing the others, so these helpers drop/replace a single layer at a
time (via raw sqlite3, never geopandas mode="w" which would wipe the file), append
a raster layer through GDAL, and rebuild the Patch_Grid annotation-status layer.

sqlite3 is standard library; gdal / geopandas / shapely are imported lazily inside
the functions that use them, so importing this module stays cheap.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from common import ROOT, rel_to_root


def layer_exists(gpkg_path: Path, layer_name: str) -> bool:
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


def drop_vector_layer(gpkg_path: Path, layer_name: str) -> None:
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


def drop_raster_layer(gpkg_path: Path, table_name: str) -> None:
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


def add_raster_layer(stitched_path: Path, gpkg_path: Path, layer_name: str) -> None:
    from osgeo import gdal
    drop_raster_layer(gpkg_path, layer_name)
    src = gdal.Open(str(stitched_path))
    if src is None:
        print(f"Warning: GDAL could not open {stitched_path} - raster layer skipped.")
        return
    gdal.Translate(str(gpkg_path), src, format="GPKG",
                   creationOptions=[f"RASTER_TABLE={layer_name}", "APPEND_SUBDATASET=YES"])
    src = None


def announce_target(out_path: Path, layers: list[str]) -> None:
    """Print the output GeoPackage and which of `layers` this run will replace."""
    print(f"Output GPKG  : {rel_to_root(out_path)}")
    existing = [n for n in layers if layer_exists(out_path, n)]
    if existing:
        print(f"  Replacing existing layer(s): {', '.join(existing)}")


def write_patch_grid(gpkg_path: Path, meta_path: Path, transform,
                     crs, has_georef: bool, sheet_id: str, cfg: dict) -> None:
    """
    Rebuild the Patch_Grid layer every run so annotation columns stay current.

    Attribute columns:
      patch_id           - unique patch identifier
      sheet_id           - parent sheet name
      ann_<feature>      - True if an annotation mask exists for that feature/patch
      annotated_features - comma-separated list of annotated features (empty = all predicted)
    """
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    drop_vector_layer(gpkg_path, "Patch_Grid")
    if not meta_path.exists():
        print("  Patch_Grid: metadata CSV not found - skipping.")
        return

    # Discover which features have annotation masks for this sheet
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
