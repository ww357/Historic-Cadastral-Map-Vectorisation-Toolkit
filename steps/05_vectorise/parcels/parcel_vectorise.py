"""
Attribute-join and GeoPackage write for predicted parcel polygons.

Reads parcel_preds.geojson (produced by parcel_segment.py — the point-seeded
watershed step), joins all apportionment attribute columns from the original
parcel_points GeoPackage (joined on rowid), applies a minimum-area filter and
Douglas-Peucker simplification, then writes the result as a "parcels" layer in
the sheet GeoPackage.

This step is schema-driven (it only needs Features with a `rowid` property), so
it is agnostic to how the polygons were produced.

Run after parcel_segment.py has completed:
    conda activate polygons
    python steps/04_predict/parcels/parcel_segment.py   --sheet Timberscombe
    python steps/05_vectorise/parcels/parcel_vectorise.py --sheet Timberscombe
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import struct
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]

# ── PROJ database fix ─────────────────────────────────────────────────────────
# Mirror of parcel_predict.py — pyproj sometimes cannot locate proj.db in conda
# environments when pip-installed.  Set PROJ_DATA before any geo imports.
if "PROJ_DATA" not in os.environ:
    _env_root = Path(sys.executable).parents[1]
    _proj_candidates: list[Path] = [_env_root / "share" / "proj"]
    import importlib.util as _ilu
    _spec = _ilu.find_spec("pyproj")
    if _spec and _spec.submodule_search_locations:
        _pkg = Path(list(_spec.submodule_search_locations)[0])
        _proj_candidates += [
            _pkg / "proj_dir" / "share" / "proj",
            _pkg / "data",
        ]
    for _p in _proj_candidates:
        if (_p / "proj.db").exists():
            os.environ["PROJ_DATA"] = str(_p)
            break

# ── pyogrio sanity-check / eviction ──────────────────────────────────────────
# Broken pyogrio (stale PROJ, partial uninstall) causes AttributeError on import.
# Evict it so geopandas falls back to fiona.
_pyogrio_ok = False
try:
    import pyogrio as _pyogrio_test
    _ = _pyogrio_test.read_dataframe
    _pyogrio_ok = True
except ValueError as _e:
    if "DATABASE.LAYOUT.VERSION" in str(_e) or "proj_data" in str(_e).lower():
        print("Warning: pyogrio PROJ conflict — falling back to fiona.")
except (ImportError, AttributeError):
    pass

if not _pyogrio_ok:
    for _k in list(sys.modules.keys()):
        if _k == "pyogrio" or _k.startswith("pyogrio."):
            del sys.modules[_k]
    sys.modules["pyogrio"] = None  # type: ignore[assignment]

import geopandas as gpd  # noqa: E402
from shapely.geometry import shape as shapely_shape  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_geojson_to_gdf(path: Path) -> gpd.GeoDataFrame:
    """
    Read a GeoJSON file using the stdlib json module + shapely, bypassing
    geopandas / fiona / pyproj CRS resolution entirely.
    Returns a GeoDataFrame with CRS set to EPSG:27700 (all parcel predictions
    are in BNG — the TIF and GeoPackage are both EPSG:27700).

    CRS strategy: build pyproj.CRS from a fully self-contained PROJ4 string
    (explicit ellipsoid radii + towgs84 shifts) so PROJ never needs to query
    its database.  Passing the resulting CRS *object* to gdf.crs bypasses the
    second lookup that geopandas would otherwise make via from_user_input().
    """
    import pyproj

    doc = json.loads(path.read_text())
    rows: list[dict] = []
    for feat in doc.get("features", []):
        props = feat.get("properties") or {}
        geom  = feat.get("geometry")
        rows.append({**props, "geometry": shapely_shape(geom) if geom else None})

    gdf = gpd.GeoDataFrame(rows, geometry="geometry")

    # PROJ4 for OSGB36 / British National Grid — fully numeric, no authority-code lookup.
    # +a/+b replace +ellps=airy; +towgs84 replaces +datum=OSGB36.
    # PROJ can construct this entirely from built-in projection code, no proj.db needed.
    BNG_PROJ4 = (
        "+proj=tmerc +lat_0=49 +lon_0=-2 +k=0.9996012717 "
        "+x_0=400000 +y_0=-100000 "
        "+a=6377563.396 +b=6356256.909 "
        "+towgs84=446.448,-125.157,542.06,0.15,0.247,0.842,-20.489 "
        "+units=m +no_defs"
    )
    try:
        bng_crs = pyproj.CRS.from_proj4(BNG_PROJ4)
        # Assigning a pyproj.CRS object calls from_user_input(obj) internally,
        # which just copies it — no EPSG database lookup.
        gdf.crs = bng_crs
    except Exception as exc:
        print(f"  Warning: could not set CRS ({exc}); CRS will be absent from output. "
              "You can assign it manually in QGIS.")

    return gdf


def read_gpkg_attrs(path: Path) -> pd.DataFrame:
    """
    Read all attribute columns (non-geometry) from the first feature table in
    a GeoPackage using sqlite3 — no geopandas or pyproj required.
    Returns a plain DataFrame (no geometry column).
    """
    con = sqlite3.connect(str(path))
    tables = con.execute(
        "SELECT table_name FROM gpkg_contents WHERE data_type='features'"
    ).fetchall()
    if not tables:
        raise ValueError(f"No feature tables in GeoPackage: {path}")
    table_name = tables[0][0]

    geom_col = con.execute(
        "SELECT column_name FROM gpkg_geometry_columns WHERE table_name=?",
        (table_name,),
    ).fetchone()[0]

    cur = con.execute(f"SELECT * FROM [{table_name}]")
    col_names = [d[0] for d in cur.description if d[0] != geom_col]

    # Reconstruct SELECT without the geometry column
    col_sql = ", ".join(f"[{c}]" for c in col_names)
    rows = con.execute(f"SELECT {col_sql} FROM [{table_name}]").fetchall()
    con.close()

    return pd.DataFrame(rows, columns=col_names)


# ── Main ──────────────────────────────────────────────────────────────────────

def _resolve_in_dir(folder: Path, sheet_id: str, default_name: str) -> Path:
    """Prefer a *.gpkg in `folder` whose name contains the sheet; else the default."""
    if folder.exists():
        matches = sorted(p for p in folder.glob("*.gpkg")
                         if sheet_id.lower() in p.stem.lower())
        if matches:
            return matches[0]
    return folder / default_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join attributes and write parcels layer to GeoPackage."
    )
    parser.add_argument("--sheet", required=True,
                        help="Sheet ID (must match a completed parcel_segment.py run)")
    parser.add_argument("--no-mended", action="store_true",
                        help="Write to data/outputs/<sheet>.gpkg instead of the mended GeoPackage")
    parser.add_argument("--output", default=None,
                        help="Explicit output GeoPackage path (overrides default/mended resolution)")
    args = parser.parse_args()
    sheet_id = args.sheet

    cfg   = yaml.safe_load((ROOT / "config.yaml").read_text())
    paths = cfg["paths"]
    pcfg  = cfg.get("parcels", {})

    min_area     = float(pcfg.get("min_area",           50.0))
    simplify_tol = float(pcfg.get("simplify_tolerance",  1.0))
    points_file  = pcfg.get("points_file", "Holnicote Apportionment Points.gpkg")
    mended_dir   = ROOT / pcfg.get("mended_dir", "data/mended outputs")

    pred_geojson = (ROOT / paths["predictions"]
                    / "parcels" / sheet_id / "parcel_preds.geojson")
    # Match parcel_segment.py: prefer the sheet-specific points file for the
    # attribute join (e.g. "Porlock Points.gpkg"), else the configured default.
    points_path  = _resolve_in_dir(ROOT / paths["parcel_points"], sheet_id, points_file)

    # Output target: write the `parcels` layer INTO the hand-corrected GeoPackage
    # when one exists, so it sits alongside the mended boundary/text/Patch_Grid
    # layers (ogr2ogr -update adds only this layer and preserves the others).
    # Falls back to data/outputs/<sheet>.gpkg.
    if args.output:
        gpkg_path = Path(args.output)
        if not gpkg_path.is_absolute():
            gpkg_path = ROOT / gpkg_path
    elif not args.no_mended and _resolve_in_dir(mended_dir, sheet_id, f"{sheet_id}.gpkg").exists():
        gpkg_path = _resolve_in_dir(mended_dir, sheet_id, f"{sheet_id}.gpkg")
    else:
        gpkg_path = ROOT / paths["outputs"] / f"{sheet_id}.gpkg"

    if not pred_geojson.exists():
        sys.exit(
            f"Predictions not found: {pred_geojson}\n"
            f"Run parcel_predict.py --sheet {sheet_id} first."
        )
    if not points_path.exists():
        sys.exit(f"Parcel points file not found: {points_path}")

    gpkg_path.parent.mkdir(parents=True, exist_ok=True)

    is_mended = mended_dir in gpkg_path.parents
    print(f"Points join : {points_path.name}")
    print(f"Output GPKG : {gpkg_path.relative_to(ROOT)}"
          f"{'   (MENDED — parcels layer added, other layers preserved)' if is_mended else ''}")

    # ── Load predictions (json + shapely — no pyproj CRS lookup) ─────────────
    print(f"Reading {pred_geojson.name} ...")
    gdf = read_geojson_to_gdf(pred_geojson)
    print(f"  {len(gdf):,} raw parcel polygons   CRS: {gdf.crs}")

    # ── Min-area filter ───────────────────────────────────────────────────────
    if min_area > 0:
        before = len(gdf)
        gdf = gdf[gdf.geometry.area >= min_area].copy()
        removed = before - len(gdf)
        print(f"  Min-area filter ({min_area} m²): removed {removed}, kept {len(gdf)}")

    # ── Simplify (topology-preserving across the whole coverage) ──────────────
    # A plain per-polygon .simplify() moves each shared edge independently, which
    # re-opens slivers/overlaps between neighbouring parcels.  shapely.coverage_simplify
    # simplifies the shared edge network ONCE, so adjacent parcels stay joined.
    # Fall back to leaving edges unsimplified (gap-free but blocky) if unavailable.
    if simplify_tol > 0:
        try:
            import shapely
            simplified = shapely.coverage_simplify(gdf.geometry.values, simplify_tol)
            gdf["geometry"] = list(simplified)
            print(f"  Coverage-simplified (tolerance={simplify_tol} m; shared edges preserved)")
        except Exception as exc:
            print(f"  coverage_simplify unavailable ({type(exc).__name__}: {exc}); "
                  "leaving edges unsimplified to avoid reintroducing slivers")

    # ── Attribute join (sqlite3 — no pyproj) ──────────────────────────────────
    print(f"Reading {points_path.name} for attribute join ...")
    pts_attrs = read_gpkg_attrs(points_path)

    # Normalise rowid to nullable Int64 on both sides for a clean join.
    # rowid is per-parish in the source GeoPackage so duplicates exist across
    # the full 6003-row table.  Deduplicate pts_attrs so the left join stays
    # 1:1 (one prediction row → at most one attribute row).
    gdf["_join_id"]       = pd.to_numeric(gdf["rowid"],          errors="coerce").astype("Int64")
    pts_attrs["_join_id"] = pd.to_numeric(pts_attrs.get("rowid", pd.Series(dtype=object)),
                                          errors="coerce").astype("Int64")

    # Drop "rowid" from pts_attrs to avoid duplicate column after merge
    pts_attrs = pts_attrs.drop(columns=["rowid"], errors="ignore")

    # CRITICAL: remove duplicate join keys before merge to prevent row explosion.
    n_before_dedup = len(pts_attrs)
    pts_attrs = pts_attrs.drop_duplicates(subset=["_join_id"], keep="first")
    if len(pts_attrs) < n_before_dedup:
        print(f"  Note: deduplicated pts_attrs from {n_before_dedup} → "
              f"{len(pts_attrs)} rows on rowid (non-unique keys in source GeoPackage)")

    attr_cols   = [c for c in pts_attrs.columns if c != "_join_id"]
    before_join = len(gdf)
    gdf = gdf.merge(pts_attrs, on="_join_id", how="left", suffixes=("", "_pt"))
    gdf = gdf.drop(columns=["_join_id"])

    # Sanity check — if the merge still exploded despite dedup, abort early
    if len(gdf) > before_join * 2:
        sys.exit(
            f"ERROR: merge produced {len(gdf):,} rows from {before_join:,} predictions — "
            "the rowid join key is still non-unique in pts_attrs. "
            "Check the source GeoPackage for duplicate rowid values."
        )

    matched = gdf["ParcelID"].notna().sum() if "ParcelID" in gdf.columns else "?"
    print(f"  Joined {len(attr_cols)} attribute columns  "
          f"({matched}/{before_join} rows matched)")

    # ── Fiona compatibility: cast pandas StringDtype → plain object ───────────
    str_cols = gdf.select_dtypes(include="string").columns.tolist()
    if str_cols:
        gdf[str_cols] = gdf[str_cols].astype(object)

    # ── Rename reserved GDAL column names ────────────────────────────────────
    # GDAL treats a GeoJSON property named "fid" (or "id") as the OGR feature
    # ID and uses it as the GPKG primary key.  If two predictions inherited the
    # same fid from the attribute join (because they shared a non-unique rowid),
    # ogr2ogr fails with "UNIQUE constraint failed: parcels.fid".
    # Rename to "pt_fid" so GDAL treats it as a plain attribute column.
    rename_map = {c: f"pt_{c}" for c in ("fid",) if c in gdf.columns}
    if rename_map:
        gdf = gdf.rename(columns=rename_map)

    # ── Write via temp GeoJSON → ogr2ogr → GPKG ──────────────────────────────
    # fiona/geopandas.to_file() fails because GDAL can't build a SpatialReference
    # from our pyproj-based CRS object (same broken proj.db, different callsite).
    # ogr2ogr is a subprocess that runs in the conda env's full GDAL context
    # (GDAL_DATA / PROJ_DATA env vars set by conda activate), so EPSG:27700
    # resolves correctly there even though the Python-side pyproj is broken.
    import subprocess
    import tempfile

    print(f"Writing to {gpkg_path.name} (ogr2ogr) ...")

    # Serialise GeoDataFrame to temp GeoJSON using stdlib json — no fiona/pyproj.
    geom_col  = gdf.geometry.name
    attr_cols = [c for c in gdf.columns if c != geom_col]

    def _json_val(v):
        """Convert a pandas/numpy value to a JSON-safe Python scalar."""
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(v, (int, float, str, bool, type(None))):
            return v
        return str(v)   # fallback for any exotic dtype

    features_out = []
    for _, row in gdf.iterrows():
        geom = row[geom_col]
        if geom is None or geom.is_empty:
            continue
        features_out.append({
            "type"      : "Feature",
            "geometry"  : geom.__geo_interface__,
            "properties": {c: _json_val(row[c]) for c in attr_cols},
        })

    tmp = Path(tempfile.mktemp(suffix=".geojson"))
    tmp.write_text(json.dumps({"type": "FeatureCollection", "features": features_out},
                               separators=(",", ":")))

    # ── Scrub any existing 'parcels' layer cleanly before ogr2ogr ────────────
    # We must remove ALL artefacts of a previous parcels layer or ogr2ogr will
    # hit broken state.  The critical ones that were previously missed are the
    # R-tree spatial index tables (rtree_parcels_geom*) — if these are left
    # orphaned, ogr2ogr silently reuses them and QGIS ends up with a corrupt
    # index: features disappear at zoom levels and can't be selected.
    if gpkg_path.exists():
        _con = sqlite3.connect(str(gpkg_path))
        try:
            # 1. Drop all triggers referencing parcels
            _triggers = _con.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='trigger' AND name LIKE '%parcels%'"
            ).fetchall()
            for (_t,) in _triggers:
                _con.execute(f"DROP TRIGGER IF EXISTS [{_t}]")

            # 2. Drop R-tree spatial index tables (geometry column is 'geom')
            for _sfx in ("", "_rowid", "_node", "_parent"):
                _con.execute(f"DROP TABLE IF EXISTS [rtree_parcels_geom{_sfx}]")

            # 3. Remove from GeoPackage metadata / extension tables
            _con.execute("DELETE FROM gpkg_contents         WHERE table_name='parcels'")
            _con.execute("DELETE FROM gpkg_geometry_columns WHERE table_name='parcels'")
            for _ext_tbl in ("gpkg_ogr_contents", "gpkg_extensions",
                             "gpkg_metadata_reference"):
                try:
                    _con.execute(f"DELETE FROM [{_ext_tbl}] WHERE table_name='parcels'")
                except Exception:
                    pass   # table may not exist in all GPKG versions

            # 4. Drop the feature table itself
            _con.execute("DROP TABLE IF EXISTS [parcels]")
            _con.commit()
        except Exception as _e:
            print(f"  (pre-clean note: {_e})")
        finally:
            _con.close()

    # Build ogr2ogr command — uses GDAL's own EPSG database, not pyproj.
    # -update adds the new layer to the existing GPKG (preserving other layers).
    # -overwrite is NOT used here because we already deleted the layer above;
    # using it on a cleanly-removed layer causes the fid UNIQUE constraint error.
    cmd = [
        "ogr2ogr",
        "-f", "GPKG",
        str(gpkg_path),
        str(tmp),
        "-nln", "parcels",
        "-a_srs", "EPSG:27700",
        "-nlt", "POLYGON",
        "-update",              # append to existing GPKG (keeps other layers)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(
                f"ogr2ogr failed (exit {result.returncode}):\n"
                f"{result.stderr.strip()}\n\n"
                "Make sure ogr2ogr is on PATH:  which ogr2ogr"
            )
    finally:
        tmp.unlink(missing_ok=True)

    print(f"\nDone — {len(features_out):,} parcels in 'parcels' layer of {gpkg_path.name}")


if __name__ == "__main__":
    main()
