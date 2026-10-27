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
import sys
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


# -- Manual layers -----------------------------------------------------------------
# Layers imported from data/manual_layers/ (import_layers.py) are the user's own finished
# work, not model output. Every feature carries source = "manual"; that column is the mark
# (it survives being copied into a mended GeoPackage or edited in QGIS).

MANUAL_TAG = "manual"
_replace_manual = False


def allow_manual_replace(flag: bool) -> None:
    """Set from a step's --replace-manual flag: lets a model-output step overwrite a manual layer."""
    global _replace_manual
    _replace_manual = bool(flag)


def manual_layers(gpkg_path: Path) -> list[str]:
    """Vector layers in `gpkg_path` whose features are tagged source = 'manual' (sqlite only)."""
    if not gpkg_path or not Path(gpkg_path).exists():
        return []
    found = []
    con = sqlite3.connect(gpkg_path)
    try:
        tables = [r[0] for r in con.execute("SELECT table_name FROM gpkg_contents WHERE data_type='features'")]
        for t in tables:
            if t == "Patch_Grid":
                continue
            cols = [r[1] for r in con.execute(f"PRAGMA table_info([{t}])")]
            if "source" in cols and con.execute(
                    f"SELECT 1 FROM [{t}] WHERE source = ? LIMIT 1", (MANUAL_TAG,)).fetchone():
                found.append(t)
    except sqlite3.Error:
        pass
    finally:
        con.close()
    return found


def guard_manual(out_path: Path, layers: list[str]) -> None:
    """Exit rather than overwrite a manual layer, unless --replace-manual was given."""
    if _replace_manual:
        return
    manual = set(manual_layers(out_path))
    clash = [n for n in layers if n in manual]
    if clash:
        sys.exit(
            f"Refusing to overwrite manual layer(s): {', '.join(clash)}\n"
            f"  in {rel_to_root(out_path)}. They were imported from data/manual_layers/ (features tagged "
            f"source='manual'), so they are your own work, not model output.\n"
            f"  Pass --replace-manual to replace them with this step's output anyway.")


def announce_target(out_path: Path, layers: list[str]) -> None:
    """Print the output GeoPackage and which of `layers` this run will replace.
    Refuses (exits) if one of them is a manual layer and --replace-manual was not given."""
    print(f"Output GPKG  : {rel_to_root(out_path)}")
    guard_manual(out_path, layers)
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
      manual_<layer>     - True if the patch touches a manual layer (source = 'manual')
      manual_features    - comma-separated manual layers touching the patch
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

    # Manual layers (imported by import_layers.py): flag every patch the layer touches, the
    # same way ann_<feature> flags annotated patches.
    manual = manual_layers(gpkg_path)
    # Tiny inward shrink so a feature that merely runs along a shared patch edge does not
    # light up the neighbouring patch too.
    inner = gpd.GeoDataFrame({"patch_id": grid_gdf["patch_id"]},
                             geometry=grid_gdf.geometry.buffer(-0.01), crs=grid_gdf.crs)
    for name in manual:
        lyr = gpd.read_file(gpkg_path, layer=name, columns=[])          # geometry only
        hit = gpd.sjoin(inner, lyr[["geometry"]], how="inner", predicate="intersects")["patch_id"]
        grid_gdf[f"manual_{name}"] = grid_gdf["patch_id"].isin(set(hit))
    if manual:
        grid_gdf["manual_features"] = [
            ", ".join(n for n in manual if grid_gdf.at[i, f"manual_{n}"]) for i in grid_gdf.index]

    write_mode = "a" if gpkg_path.exists() else "w"
    grid_gdf.to_file(gpkg_path, driver="GPKG", layer="Patch_Grid", mode=write_mode)
    ann_cols = [f"ann_{f}" for f in feature_mask_dirs] or ["(none)"]
    man_cols = [f"manual_{n}" for n in manual]
    print(f"  Patch_Grid (vector):  {len(grid_gdf):,} patches  |  "
          f"annotation columns: {', '.join(ann_cols)}"
          + (f"  |  manual columns: {', '.join(man_cols)}" if man_cols else ""))
