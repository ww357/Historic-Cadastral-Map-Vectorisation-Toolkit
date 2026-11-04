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

import hashlib
import os
import sqlite3
import sys
from datetime import datetime
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
        try:
            con.execute(f"DELETE FROM {STATE_TABLE} WHERE layer = ?", (layer_name,))   # stale stamp
        except sqlite3.OperationalError:
            pass                                                                      # no state table yet
        con.commit()
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower():
            sys.exit(f"{gpkg_path.name} is locked by another program (QGIS?), so layer '{layer_name}' "
                     f"could not be replaced. Close it there and re-run.")
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


# -- Layer state: manual / edited / backups / open in another program ---------------------
#
# Users correct layers in QGIS directly inside the output GeoPackage. Without a record of
# what the toolkit wrote, a later vectorise run would silently overwrite those corrections,
# and the feedback / parcel steps could not tell a mended layer from raw model output.
#
# Two marks, both plain sqlite (no geopandas, so status.py can use them):
#   manual layer - every feature carries source = 'manual' (import_layers.py).
#   layer stamp  - table toolkit_layer_state(layer, written_at, fingerprint, n_features),
#                  written right after a step writes a layer. The fingerprint is a hash of
#                  the layer's GEOMETRIES (order-independent), so adding, deleting or moving
#                  features changes it; attribute-only edits deliberately do not.
# A layer is "edited" when its fingerprint no longer matches its stamp, and "untracked" when
# it has no stamp (written before this tracking existed, or by another program) - which is
# treated as possibly edited, because we cannot prove it is not.

MANUAL_TAG = "manual"
STATE_TABLE = "toolkit_layer_state"
KEEP_BACKUPS = 5                      # newest backups kept per GeoPackage
_flags = {"replace_manual": False, "replace_edited": False, "ignore_open": False}


def add_safety_flags(parser, manual: bool = True) -> None:
    """Add the shared overwrite-safety flags to a step's argparse parser
    (manual=False omits --replace-manual, which only model-output steps need)."""
    g = parser.add_argument_group("overwrite safety")
    g.add_argument("--replace-edited", action="store_true",
                   help="Overwrite a layer you have edited (or that has no edit record). "
                        "A backup copy of the GeoPackage is saved first.")
    if manual:
        g.add_argument("--replace-manual", action="store_true",
                       help="Allow overwriting a manual layer (imported with import-manual) "
                            "with this step's model output.")
    g.add_argument("--ignore-open", action="store_true",
                   help="Carry on even if another program (QGIS) has the GeoPackage open. "
                        "Risky: unsaved edits in QGIS can be lost.")


def apply_safety_flags(args) -> None:
    """Call right after parse_args() (after add_safety_flags)."""
    for key in _flags:
        _flags[key] = bool(getattr(args, key, False))


def vector_layers(gpkg_path: Path) -> list[str]:
    """Vector (feature) layers in the GeoPackage, excluding the toolkit's Patch_Grid."""
    if not gpkg_path or not Path(gpkg_path).exists():
        return []
    con = sqlite3.connect(gpkg_path)
    try:
        return [r[0] for r in con.execute(
            "SELECT table_name FROM gpkg_contents WHERE data_type='features'") if r[0] != "Patch_Grid"]
    except sqlite3.Error:
        return []
    finally:
        con.close()


def manual_layers(gpkg_path: Path) -> list[str]:
    """Vector layers whose features are tagged source = 'manual'."""
    found = []
    if not gpkg_path or not Path(gpkg_path).exists():
        return found
    con = sqlite3.connect(gpkg_path)
    try:
        for t in vector_layers(gpkg_path):
            cols = [r[1] for r in con.execute(f"PRAGMA table_info([{t}])")]
            if "source" in cols and con.execute(
                    f"SELECT 1 FROM [{t}] WHERE source = ? LIMIT 1", (MANUAL_TAG,)).fetchone():
                found.append(t)
    except sqlite3.Error:
        pass
    finally:
        con.close()
    return found


def _wkb_of(blob) -> bytes:
    """Geometry bytes without the GeoPackage header, so a re-encoded header (envelope, flags)
    is not mistaken for an edit."""
    if not blob or len(blob) < 8 or blob[:2] != b"GP":
        return bytes(blob or b"")
    env = (blob[3] >> 1) & 0x07
    return bytes(blob[8 + {1: 32, 2: 48, 3: 48, 4: 64}.get(env, 0):])


def layer_fingerprint(gpkg_path: Path, layer: str) -> tuple[str, int] | None:
    """(hash of all geometries, feature count) for a vector layer; None if unreadable."""
    con = sqlite3.connect(gpkg_path, timeout=5)
    try:
        row = con.execute("SELECT column_name FROM gpkg_geometry_columns WHERE table_name = ?",
                          (layer,)).fetchone()
        if not row:
            return None
        digests = sorted(hashlib.blake2b(_wkb_of(b), digest_size=8).digest()
                         for (b,) in con.execute(f"SELECT [{row[0]}] FROM [{layer}]"))
        return hashlib.blake2b(b"".join(digests), digest_size=16).hexdigest(), len(digests)
    except sqlite3.Error:
        return None
    finally:
        con.close()


def stamp_layer(gpkg_path: Path, layer: str) -> None:
    """Record that the toolkit just wrote `layer`, so later edits can be detected.
    Call immediately after writing the layer."""
    fp = layer_fingerprint(gpkg_path, layer)
    if fp is None:
        return
    con = sqlite3.connect(gpkg_path, timeout=10)
    try:
        con.execute(f"CREATE TABLE IF NOT EXISTS {STATE_TABLE} "
                    "(layer TEXT PRIMARY KEY, written_at TEXT, fingerprint TEXT, n_features INTEGER)")
        con.execute(f"INSERT OR REPLACE INTO {STATE_TABLE} VALUES (?, ?, ?, ?)",
                    (layer, datetime.now().isoformat(timespec="seconds"), fp[0], fp[1]))
        con.commit()
    except sqlite3.Error as e:
        print(f"  Note: could not record the edit-tracking stamp for '{layer}' ({e})")
    finally:
        con.close()


def layer_edit_state(gpkg_path: Path, layer: str) -> str:
    """'missing' | 'untracked' (no stamp) | 'unchanged' (as the toolkit wrote it) | 'edited'
    (features added, deleted or moved since). Attribute-only edits do not count."""
    if layer not in vector_layers(gpkg_path):
        return "missing"
    con = sqlite3.connect(gpkg_path, timeout=5)
    try:
        row = con.execute(f"SELECT fingerprint FROM {STATE_TABLE} WHERE layer = ?", (layer,)).fetchone()
    except sqlite3.OperationalError:                      # no state table yet
        row = None
    finally:
        con.close()
    if row is None:
        return "untracked"
    fp = layer_fingerprint(gpkg_path, layer)
    return "unchanged" if fp is not None and fp[0] == row[0] else "edited"


# -- Backups -----------------------------------------------------------------------------

def backup_gpkg(gpkg_path: Path) -> Path:
    """Copy the GeoPackage to <its folder>/backups/<name>_<timestamp>.gpkg using sqlite's backup
    API (consistent even if the file is mid-use), and keep only the newest KEEP_BACKUPS."""
    gpkg_path = Path(gpkg_path)
    folder = gpkg_path.parent / "backups"
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{gpkg_path.stem}_{datetime.now():%Y%m%d-%H%M%S}.gpkg"
    src, dst = sqlite3.connect(gpkg_path, timeout=10), sqlite3.connect(dest)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    for old in sorted(folder.glob(f"{gpkg_path.stem}_*.gpkg"))[:-KEEP_BACKUPS]:
        old.unlink(missing_ok=True)
    return dest


# -- Is another program (QGIS) using the file? ------------------------------------------------

def open_elsewhere(gpkg_path: Path) -> list[str]:
    """Names of OTHER processes holding the GeoPackage (or its -wal file) open. Linux/WSL only
    (reads /proc) - which covers QGIS run inside WSL. Returns [] where it cannot tell."""
    if not sys.platform.startswith("linux"):
        return []
    targets = {os.path.realpath(gpkg_path), os.path.realpath(str(gpkg_path) + "-wal")}
    me, found = os.getpid(), []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == me:
            continue
        try:
            for fd in (proc / "fd").iterdir():
                try:
                    if os.path.realpath(os.readlink(fd)) in targets:
                        name = (proc / "comm").read_text().strip()
                        found.append(f"{name} (pid {proc.name})")
                        break
                except OSError:
                    continue
        except OSError:
            continue                                      # not ours / already gone
    return found


def ensure_not_open(gpkg_path: Path) -> None:
    """Before a step REWRITES layers: refuse if another program has the file open, or holds a
    write lock. QGIS keeps unsaved edits in memory, so rewriting the tables underneath it can
    lose them or leave QGIS showing stale layers."""
    gpkg_path = Path(gpkg_path)
    if not gpkg_path.exists():
        return
    holders = open_elsewhere(gpkg_path)
    if holders and not _flags["ignore_open"]:
        sys.exit(
            f"{gpkg_path.name} is open in another program: {', '.join(holders)}.\n"
            f"  Save your edits, then close the project (or remove the layers) in QGIS and re-run.\n"
            f"  Rewriting layers under an open QGIS can lose unsaved edits.\n"
            f"  (--ignore-open skips this check - not recommended.)")
    con = None
    try:
        con = sqlite3.connect(gpkg_path, timeout=1)
        con.execute("BEGIN IMMEDIATE")                    # can we take the write lock right now?
        con.rollback()
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower():
            sys.exit(f"{gpkg_path.name} is locked by another program that is writing to it "
                     f"(QGIS saving?). Wait for it to finish, close it, and re-run.")
    finally:
        if con is not None:
            con.close()
    wal = Path(str(gpkg_path) + "-wal")
    if not holders and wal.exists() and wal.stat().st_size > 0:
        print(f"  Note: {wal.name} exists, so a program may still have {gpkg_path.name} open. "
              f"If QGIS is running, close it first.")


def warn_if_open(gpkg_path: Path) -> None:
    """For steps that only READ the GeoPackage: reading is safe, but edits not yet saved in
    QGIS are not in the file, so say so."""
    holders = open_elsewhere(Path(gpkg_path)) if Path(gpkg_path).exists() else []
    if holders:
        print(f"\n  ! {Path(gpkg_path).name} is open in {', '.join(holders)}.\n"
              f"    Edits you have not saved in QGIS are not in the file, so this step cannot see them.\n")


# -- The one check every step that rewrites layers runs ------------------------------------------

def guard_write(out_path: Path, layers: list[str], *, manual_ok: bool = False) -> None:
    """Call before dropping/rewriting `layers` in `out_path`. Exits (with a plain explanation)
    if the file is open elsewhere, if a manual layer would be overwritten, or if a layer holds
    edits (or cannot be shown to be untouched). When overwriting edited layers is explicitly
    allowed (--replace-edited) a backup copy of the whole GeoPackage is saved first.
    manual_ok=True is for import_layers.py, whose job is to (re)write manual layers."""
    out_path = Path(out_path)
    if not out_path.exists():
        return
    ensure_not_open(out_path)

    present = set(vector_layers(out_path))
    target = [n for n in layers if n in present]
    manual = set(manual_layers(out_path))

    clash = [n for n in target if n in manual]
    if clash and not manual_ok and not _flags["replace_manual"]:
        sys.exit(
            f"Refusing to overwrite manual layer(s): {', '.join(clash)}\n"
            f"  in {rel_to_root(out_path)}. They were imported from data/manual_layers/ (features tagged "
            f"source='manual'), so they are your own work, not model output.\n"
            f"  Pass --replace-manual to replace them with this step's output anyway.")

    reasons = {"edited": "edited since the toolkit wrote it (features added, deleted or moved)",
               "untracked": "no edit record (written before edit tracking existed), so it cannot be "
                            "shown to be untouched"}
    protected = [(n, layer_edit_state(out_path, n)) for n in target]
    protected = [(n, s) for n, s in protected if s in reasons]
    if not protected:
        return
    if not _flags["replace_edited"]:
        lines = "\n".join(f"    {n:<16} {reasons[s]}" for n, s in protected)
        sys.exit(
            f"Refusing to overwrite layer(s) that may hold your edits in {rel_to_root(out_path)}:\n{lines}\n"
            f"  Re-run with --replace-edited to replace them anyway - a backup copy of the GeoPackage "
            f"is saved first.\n"
            f"  (Other layers are unaffected; a step only ever rewrites its own layers.)")
    saved = backup_gpkg(out_path)
    print(f"  Backup of the GeoPackage saved: {rel_to_root(saved)}  (before replacing: "
          f"{', '.join(n for n, _ in protected)})")


def announce_target(out_path: Path, layers: list[str]) -> None:
    """Print the output GeoPackage and which of `layers` this run will replace, after the
    safety checks in guard_write (open elsewhere / manual / edited layers)."""
    print(f"Output GPKG  : {rel_to_root(out_path)}")
    guard_write(out_path, layers)
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
