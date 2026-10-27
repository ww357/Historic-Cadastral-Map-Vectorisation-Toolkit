"""
Import fully hand-made vector layers (made in QGIS etc.) into the sheet's GeoPackage.

Put finished layers in  data/manual_layers/<SHEET_ID>/  - the FILE NAME is the layer name:
    houses.shp      -> layer "houses"
    bog.geojson     -> layer "bog"
    boundaries.gpkg -> layer "boundaries"     (a "boundary" file is treated the same way)
Any vector format GDAL reads works (shp, geojson, gpkg, kml, gml, fgb, gdb, ...). A file
holding several layers (e.g. a .gpkg) gives one layer per inner layer, named after each.

Each layer is then
  - reprojected to the raster map's CRS (a file with no CRS is assumed to already be in it),
  - cleaned (empty geometries dropped, invalid ones repaired, Z removed),
  - tagged: every feature gets  sheet_id, feature, source = "manual", source_file, imported
    (your own attribute columns are kept),
  - written as a layer of the sheet's GeoPackage, and Patch_Grid is rebuilt with
    manual_<layer> columns showing which patches each manual layer touches.

The `source = "manual"` tag is what protects the layer: the model-output steps refuse to
overwrite it (unless given --replace-manual) and the feedback steps skip it.

Output target follows the toolkit-wide rule: default data/outputs/<SHEET_ID>.gpkg;
--mended writes into the hand-corrected GeoPackage; --gpkg overrides both.

Usage:
    python import_layers.py --sheet SHEET_ID [--layers houses bog] [--dry-run]
                            [--mended | --gpkg PATH] [--assume-crs EPSG:27700]
                            [--replace-existing]
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from datetime import date
from pathlib import Path

import geopandas as gpd
import pyogrio
import shapely
import yaml
from pyproj import CRS

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "steps"))   # shared helpers
from common import load_config, manual_layers_dir, rel_to_root, resolve_output_gpkg   # noqa: E402
from geopackage import MANUAL_TAG, layer_exists, manual_layers, drop_vector_layer, write_patch_grid   # noqa: E402
from vectorise_common import sheet_georef   # noqa: E402

# Files that sit beside a vector layer (shapefile parts, QGIS styles, notes) - never layers.
IGNORE_SUFFIXES = {".dbf", ".shx", ".prj", ".cpg", ".qix", ".sbn", ".sbx", ".qpj", ".qmd", ".qml",
                   ".sld", ".xml", ".lock", ".txt", ".md", ".csv", ".ini", ".db"}
RESERVED = {"patch_grid"}                                        # written by the toolkit itself
ALIASES = {"boundary": "boundaries"}                             # same layer the model writes
STANDARD_COLS = ["sheet_id", "feature", "source", "source_file", "imported"]
NOT_CLASSES = {"labelme_json", "original_weights", "feedback"}   # folders that are not feature classes


# -- Names -----------------------------------------------------------------------

def clean_name(stem: str) -> str:
    """File stem -> layer name: spaces/odd characters become '_', a 'boundary' file maps to
    the toolkit's 'boundaries' layer."""
    name = re.sub(r"[^\w.-]+", "_", stem.strip()).strip("_")
    return ALIASES.get(name.lower(), name)


def known_classes(cfg: dict) -> set[str]:
    """Class names the toolkit already uses (annotations, predictions, MapSAM adapters, labelme)."""
    names = {"boundaries", "dashed", "text", "parcels"}
    for root in (ROOT / cfg["paths"]["annotations"], ROOT / cfg["paths"]["predictions"],
                 ROOT / cfg["paths"]["models_base"] / "MapSAM"):
        if root.is_dir():
            names |= {d.name for d in root.iterdir() if d.is_dir()}
    lm = ROOT / "steps" / "02_annotate" / "labelme_config.yaml"
    if lm.exists():
        names |= set(yaml.safe_load(lm.read_text()).get("labels") or [])
    return names - NOT_CLASSES


def _forms(s: str) -> set[str]:
    s = re.sub(r"[^a-z0-9]", "", s.lower())
    return {s} | ({s[:-1]} if s.endswith("s") else set()) | ({s[:-2]} if s.endswith("es") else set())


def classify_name(name: str, classes: set[str]) -> tuple[str, str | None]:
    """('existing'|'near'|'new', closest class). 'near' = same word in another spelling
    (plural, case, small typo): it would become a separate class, so it is worth a look."""
    if name in classes:
        return "existing", name
    for c in sorted(classes):
        if c.lower() == name.lower() or _forms(c) & _forms(name):
            return "near", c
    close = difflib.get_close_matches(name.lower(), [c.lower() for c in classes], n=1, cutoff=0.8)
    if close:
        return "near", next(c for c in classes if c.lower() == close[0])
    return "new", None


# -- Reading ---------------------------------------------------------------------

def discover(folder: Path) -> tuple[list[dict], list[str]]:
    """One item per layer found; plus notes for files that were skipped."""
    items, skipped = [], []
    for p in sorted(folder.iterdir()):
        if p.name.startswith(".") or p.name.endswith(".aux.xml"):
            continue
        if p.is_dir() and p.suffix.lower() != ".gdb":
            continue
        if p.is_file() and p.suffix.lower() in IGNORE_SUFFIXES:
            continue
        try:
            inner = [str(r[0]) for r in pyogrio.list_layers(p)]
        except Exception:
            skipped.append(f"{p.name}: not a readable vector file")
            continue
        if len(inner) == 1:                                  # single layer: the FILE name is the name
            items.append({"path": p, "inner": None, "stem": p.stem})
        else:
            items += [{"path": p, "inner": n, "stem": n} for n in inner]
    return items, skipped


# -- Cleaning --------------------------------------------------------------------

def _dominant(g):
    """A repaired geometry can come back as a collection; keep its highest-dimension parts."""
    if g.geom_type != "GeometryCollection":
        return g
    top = max((p.dimension for p in g.geoms), default=-1)
    parts = [p for p in g.geoms if p.dimension == top]
    return shapely.union_all(parts) if parts else g


def clean_geometries(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict]:
    stats = {"dropped": 0, "repaired": 0, "flattened": 0}
    n0 = len(gdf)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.geometry.has_z.any():
        stats["flattened"] = int(gdf.geometry.has_z.sum())
        gdf["geometry"] = shapely.force_2d(gdf.geometry.values)
    bad = ~gdf.geometry.is_valid
    if bad.any():
        stats["repaired"] = int(bad.sum())
        gdf.loc[bad, "geometry"] = gdf.loc[bad, "geometry"].make_valid().map(_dominant)
    gdf = gdf[~gdf.geometry.is_empty]
    stats["dropped"] = n0 - len(gdf)
    return gdf, stats


def same_crs(a, b) -> bool:
    return a == b or (a.to_epsg() is not None and a.to_epsg() == b.to_epsg())


def to_map_crs(gdf: gpd.GeoDataFrame, map_crs, assume: str | None) -> tuple[gpd.GeoDataFrame, str]:
    """Put the layer in the raster map's CRS. Returns (layer, note)."""
    if map_crs is None:                                          # ungeoreferenced map: pixel coordinates
        if gdf.crs is not None:
            return gdf.set_crs(None, allow_override=True), "map has no CRS (pixel coordinates): layer CRS dropped, numbers kept as-is"
        return gdf, "pixel coordinates"
    target = CRS.from_wkt(map_crs.to_wkt()) if hasattr(map_crs, "to_wkt") else CRS.from_user_input(map_crs)
    if gdf.crs is None:
        gdf = gdf.set_crs(assume or target)
        note = f"no CRS in file - assumed {assume or target.to_string()}"
    else:
        note = f"CRS {gdf.crs.to_string()}"
    if same_crs(gdf.crs, target):
        return gdf, note + " (matches map)"
    return gdf.to_crs(target), note + f" -> reprojected to {target.to_string()}"


def tag(gdf: gpd.GeoDataFrame, sheet_id: str, name: str, source_file: str) -> gpd.GeoDataFrame:
    """Standard toolkit columns first; the user's own columns kept (renamed if they clash)."""
    gdf = gdf.rename(columns={c: f"orig_{c}" for c in gdf.columns if c in STANDARD_COLS})
    for i, (col, val) in enumerate([("sheet_id", sheet_id), ("feature", name), ("source", MANUAL_TAG),
                                    ("source_file", source_file), ("imported", date.today().isoformat())]):
        gdf.insert(i, col, val)
    str_cols = gdf.select_dtypes(include="string").columns.tolist()
    if str_cols:
        gdf[str_cols] = gdf[str_cols].astype(object)
    return gdf.reset_index(drop=True)


# -- Driver ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Import hand-made vector layers into the sheet GeoPackage.")
    ap.add_argument("--sheet", required=True, help="Sheet ID")
    ap.add_argument("--layers", nargs="+", default=None, help="Only these layer names (default: all found)")
    ap.add_argument("--dry-run", action="store_true", help="Check and report everything; write nothing")
    ap.add_argument("--assume-crs", default=None, metavar="EPSG:xxxx",
                    help="CRS to assume for files that have none (default: the map's CRS)")
    ap.add_argument("--replace-existing", action="store_true",
                    help="Also replace a layer of the same name that is model output (not manual)")
    target = ap.add_mutually_exclusive_group()
    target.add_argument("--mended", action="store_true",
                        help="Write into the hand-corrected GeoPackage in paths.outputs_mended")
    target.add_argument("--gpkg", default=None, help="Explicit output GeoPackage (overrides --mended and the default)")
    args = ap.parse_args()

    cfg, sheet = load_config(), args.sheet
    folder = manual_layers_dir(cfg, sheet)
    if not folder.is_dir():
        sys.exit(f"No manual layers folder for '{sheet}': {rel_to_root(folder)}\n"
                 f"Create it and put your layers in, e.g. {rel_to_root(folder)}/houses.shp "
                 f"(the file name becomes the layer name).")
    georef = sheet_georef(sheet, cfg)
    map_crs = georef["crs"] if georef["has_georef"] else None
    classes = known_classes(cfg)
    out_path = resolve_output_gpkg(sheet, cfg, args.gpkg, args.mended)

    items, skipped = discover(folder)
    print(f"Sheet      : {sheet}")
    print(f"Folder     : {rel_to_root(folder)}  ({len(items)} layer(s) found)")
    print(f"Map CRS    : {map_crs.to_string() if hasattr(map_crs, 'to_string') else (map_crs or 'none (pixel coordinates)')}")
    print(f"Output GPKG: {rel_to_root(out_path)}{'   [dry run - nothing will be written]' if args.dry_run else ''}\n")

    prepared, seen = [], {}
    for it in items:
        name = clean_name(it["stem"])
        if args.layers and name.lower() not in {clean_name(x).lower() for x in args.layers}:
            continue
        if name.lower() in RESERVED or not name:
            skipped.append(f"{it['path'].name}: '{name}' is a reserved name - rename the file")
            continue
        if name.lower() in seen:
            skipped.append(f"{it['path'].name}: duplicate of '{seen[name.lower()]}' (same layer name) - remove one")
            continue
        try:
            gdf = gpd.read_file(it["path"], layer=it["inner"]) if it["inner"] else gpd.read_file(it["path"])
        except Exception as e:
            skipped.append(f"{it['path'].name}: could not be read ({type(e).__name__}: {str(e)[:80]})")
            continue
        gdf, stats = clean_geometries(gdf)
        if gdf.empty:
            skipped.append(f"{it['path'].name}: no usable geometries")
            continue
        gdf, crs_note = to_map_crs(gdf, map_crs, args.assume_crs)
        gdf = tag(gdf, sheet, name, it["path"].name)
        kind, close = classify_name(name, classes)
        seen[name.lower()] = it["path"].name
        prepared.append((name, gdf, it["path"].name, crs_note, stats, kind, close))

    # -- report ------------------------------------------------------------------
    for name, gdf, fname, crs_note, stats, kind, close in prepared:
        geoms = "/".join(sorted(gdf.geom_type.unique()))
        fixes = ", ".join(f"{v} {k}" for k, v in stats.items() if v)
        print(f"  {name:<18} <- {fname:<22} {geoms:<22} {len(gdf):>7,} features")
        print(f"  {'':<18}    {crs_note}" + (f"   [{fixes}]" if fixes else ""))
        if kind == "existing":
            print(f"  {'':<18}    class '{name}' already exists in the toolkit")
        elif kind == "near":
            print(f"  {'':<18}    ! looks like the existing class '{close}'. This will be a SEPARATE class "
                  f"'{name}' - rename the file to '{close}' if you meant that one.")
        else:
            print(f"  {'':<18}    new class (nothing in the toolkit uses this name yet)")
    for s in skipped:
        print(f"  skipped: {s}")
    if not prepared:
        sys.exit("\nNothing to import.")
    if args.dry_run:
        print("\nDry run complete - nothing written.")
        return

    # -- write -------------------------------------------------------------------
    already_manual = set(manual_layers(out_path))
    written, blocked = [], []
    for name, gdf, *_ in prepared:
        if layer_exists(out_path, name) and name not in already_manual and not args.replace_existing:
            blocked.append(name)
            continue
        replacing = layer_exists(out_path, name)
        drop_vector_layer(out_path, name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(out_path, driver="GPKG", layer=name, mode="a" if out_path.exists() else "w")
        written.append(name + (" (replaced)" if replacing else ""))
    print(f"\nWrote {len(written)} layer(s) to {rel_to_root(out_path)}: {', '.join(written) or '-'}")
    if blocked:
        print(f"NOT written (a model-output layer with that name already exists): {', '.join(blocked)}\n"
              f"  Pass --replace-existing to replace it with your manual layer.")

    meta_path = ROOT / cfg["paths"]["patches"] / "metadata" / f"{sheet}_patches.csv"
    if meta_path.exists():
        write_patch_grid(out_path, meta_path, georef["transform"], georef["crs"], georef["has_georef"], sheet, cfg)
    else:
        print("  Patch_Grid not updated (no patch metadata - run patchify first).")

    print(
        f"\nNext step: open {out_path.name} in QGIS - manual layers sit beside the model layers\n"
        f"  (every feature has source = 'manual'). Optional: use some of these patches as training\n"
        f"  annotations for other maps:\n"
        f"    conda activate maptools\n"
        f"    python run.py pick-patches --sheet {sheet} --name manual\n"
        f"    python run.py export-manual --sheet {sheet} --select"
    )


if __name__ == "__main__":
    main()
