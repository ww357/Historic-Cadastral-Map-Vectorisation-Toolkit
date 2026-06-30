"""
Point-seeded watershed parcel extraction (replaces point-prompted SAM).

A tithe parcel is a *cell of a planar subdivision* defined by the surrounding
boundary lines, not an appearance object — which is why SAM (and any
single-point prompt) could never infer parcel extent.  This script instead
partitions the whole sheet:

    boundary lines  →  ridges (walls)
    apportionment centroid points  →  one seed per known parcel
    marker-controlled watershed     →  every pixel assigned to its parcel

Because each parcel is pulled out by its own seed, the boundaries do NOT need to
be topologically closed.  Where two seeded parcels are separated by a dashed or
broken line, the two flood basins simply meet at the weak ridge between them —
the seeds supply the closure that the ink lacks.  This is the key reason the
gaps/dashes that defeated strict polygonisation are tolerable here.

Inputs (all already produced by earlier pipeline steps):
    data/stitched/boundaries/<sheet>.tif      — full-sheet boundary raster (lines step)
    data/parcel_points/<points_file>          — apportionment centroid points (GeoPackage)
    data/map_area_masks/<sheet>/<sheet>.png   — optional map-area mask

Output (schema matches the old SAM step, so 05_vectorise/parcels works unchanged):
    data/predictions/parcels/<sheet>/parcel_preds.geojson   — one Polygon per parcel, with rowid
    data/predictions/parcels/<sheet>/parcel_segment_preview.png  — quick visual check

Usage:
    conda activate polygons        # (or lines) — needs scikit-image, scipy, rasterio
    python steps/04_predict/parcels/parcel_segment.py --sheet Timberscombe

Then:
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

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]

# ── PROJ database fix (mirror of the SAM parcel scripts) ───────────────────────
# pyproj can fail to locate proj.db in pip-installed conda envs.  We never need
# CRS *resolution* here (everything is EPSG:27700 and we read coords via WKB),
# but rasterio still imports fine; this guard is kept for parity/safety.
if "PROJ_DATA" not in os.environ:
    _env_root = Path(sys.executable).parents[1]
    _cands = [_env_root / "share" / "proj"]
    import importlib.util as _ilu
    _spec = _ilu.find_spec("pyproj")
    if _spec and _spec.submodule_search_locations:
        _pkg = Path(list(_spec.submodule_search_locations)[0])
        _cands += [_pkg / "proj_dir" / "share" / "proj", _pkg / "data"]
    for _p in _cands:
        if (_p / "proj.db").exists():
            os.environ["PROJ_DATA"] = str(_p)
            break

try:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.features import shapes as rio_shapes
except ImportError:
    sys.exit("rasterio is required:  conda install -c conda-forge rasterio")

try:
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed
except ImportError:
    sys.exit("scikit-image + scipy required:  pip install scikit-image scipy")

try:
    from shapely import wkb as shapely_wkb   # geometry parsing only — no PROJ/CRS
except ImportError:
    shapely_wkb = None


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    p = ROOT / "config.yaml"
    if not p.exists():
        sys.exit(f"config.yaml not found: {p}")
    return yaml.safe_load(p.read_text())


def resolve_points_file(pts_dir: Path, sheet: str, default_name: str) -> Path:
    """
    Prefer a GeoPackage in pts_dir whose filename contains the sheet name (e.g.
    'StokePero_points.gpkg'), so per-sheet point files are picked up automatically.
    Falls back to the configured default ('Holnicote Apportionment Points.gpkg').
    """
    if pts_dir.exists():
        matches = sorted(p for p in pts_dir.glob("*.gpkg")
                         if sheet.lower() in p.stem.lower())
        if matches:
            return matches[0]
    return pts_dir / default_name


# ── Apportionment points (sqlite3 + WKB — no pyproj/geopandas) ─────────────────

def read_gpkg_points_wkb(path: Path) -> list[dict]:
    """
    Decode point features from a GeoPackage via sqlite3 + WKB.
    Returns a list of dicts with all attribute columns plus _geom_x, _geom_y.
    (Copied from the SAM parcel_predict to stay dependency-light.)
    """
    con = sqlite3.connect(str(path))
    tables = con.execute(
        "SELECT table_name FROM gpkg_contents WHERE data_type='features'"
    ).fetchall()
    if not tables:
        raise ValueError(f"No feature tables found in GeoPackage: {path}")
    table_name = tables[0][0]
    geom_col = con.execute(
        "SELECT column_name FROM gpkg_geometry_columns WHERE table_name=?",
        (table_name,),
    ).fetchone()[0]

    cur = con.execute(f"SELECT * FROM [{table_name}]")
    col_names = [d[0] for d in cur.description]
    all_rows  = cur.fetchall()
    con.close()

    records: list[dict] = []
    for row in all_rows:
        row_dict  = dict(zip(col_names, row))
        geom_blob = row_dict.pop(geom_col, None)
        if not geom_blob or len(geom_blob) < 29:
            continue
        flags     = geom_blob[3]
        env_type  = (flags >> 1) & 0x07
        env_bytes = [0, 32, 48, 48, 64]
        wkb_start = 8 + (env_bytes[env_type] if env_type < 5 else 0)
        wkb       = geom_blob[wkb_start:]
        endian    = "<" if wkb[0] == 1 else ">"
        raw_type  = struct.unpack(endian + "I", wkb[1:5])[0]
        if (raw_type & 0xFFFF) not in (1, 1001, 2001, 3001):
            continue
        x, y = struct.unpack(endian + "dd", wkb[5:21])
        row_dict["_geom_x"] = x
        row_dict["_geom_y"] = y
        records.append(row_dict)
    return records


# ── Mended boundary GeoPackage → rasterised line network ──────────────────────

def resolve_mended(mended_dir: Path, sheet: str) -> Path | None:
    """Return a hand-corrected boundary GeoPackage for the sheet, or None."""
    if not mended_dir.exists():
        return None
    exact = mended_dir / f"{sheet}.gpkg"
    if exact.exists():
        return exact
    matches = sorted(p for p in mended_dir.glob("*.gpkg")
                     if sheet.lower() in p.stem.lower())
    return matches[0] if matches else None


def read_gpkg_lines_wkb(path: Path, layer: str = "boundaries") -> list[np.ndarray]:
    """
    Read LINESTRING / MULTILINESTRING geometries from a GeoPackage layer as a list
    of (N, 2) world-coordinate arrays.  Strips the GPKG geometry header, then uses
    shapely to parse the standard WKB (handles Z/M and multi-parts robustly).
    """
    if shapely_wkb is None:
        sys.exit("shapely is required to read mended boundary GeoPackages.")
    con = sqlite3.connect(str(path))
    # Pick the requested layer if present, else the first LINESTRING feature table.
    feats = [r[0] for r in con.execute(
        "SELECT table_name FROM gpkg_contents WHERE data_type='features'")]
    table = layer if layer in feats else None
    if table is None:
        for t in feats:
            gt = con.execute("SELECT geometry_type_name FROM gpkg_geometry_columns "
                             "WHERE table_name=?", (t,)).fetchone()
            if gt and "LINE" in gt[0].upper():
                table = t
                break
    if table is None:
        con.close()
        raise ValueError(f"No LINESTRING layer found in {path.name}")
    geom_col = con.execute("SELECT column_name FROM gpkg_geometry_columns "
                           "WHERE table_name=?", (table,)).fetchone()[0]
    blobs = [r[0] for r in con.execute(f"SELECT [{geom_col}] FROM [{table}]")]
    con.close()

    lines: list[np.ndarray] = []
    for blob in blobs:
        if not blob or len(blob) < 8:
            continue
        env_type  = (blob[3] >> 1) & 0x07
        env_bytes = [0, 32, 48, 48, 64]
        wkb_start = 8 + (env_bytes[env_type] if env_type < 5 else 0)
        try:
            geom = shapely_wkb.loads(bytes(blob[wkb_start:]))
        except Exception:
            continue
        parts = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
        for part in parts:
            xy = np.asarray(part.coords, dtype=np.float64)
            if xy.ndim == 2 and len(xy) >= 2:
                lines.append(xy[:, :2])
    return lines


def rasterize_lines(lines: list[np.ndarray], transform: "Affine",
                    H: int, W: int, width: int) -> np.ndarray:
    """Draw world-coordinate polylines onto an (H, W) uint8 canvas (255 = boundary)."""
    inv = ~transform                       # world → pixel affine
    canvas = np.zeros((H, W), dtype=np.uint8)
    for xy in lines:
        cols = inv.a * xy[:, 0] + inv.b * xy[:, 1] + inv.c
        rows = inv.d * xy[:, 0] + inv.e * xy[:, 1] + inv.f
        pts  = np.column_stack([cols, rows]).round().astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=False, color=255,
                      thickness=max(1, width))
    return canvas


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Point-seeded watershed parcel extraction.")
    ap.add_argument("--sheet", required=True, help="Sheet ID")
    ap.add_argument("--boundary", default=None,
                    help="Override boundary raster path (default: data/stitched/boundaries/<sheet>.tif)")
    ap.add_argument("--no-mended", action="store_true",
                    help="Ignore hand-corrected boundary GeoPackages in the mended folder")
    args = ap.parse_args()
    sheet = args.sheet

    cfg   = load_config()
    paths = cfg["paths"]
    pcfg  = cfg.get("parcels", {})

    sigma       = float(pcfg.get("boundary_smooth_sigma", 2.0))
    close_px    = int(pcfg.get("boundary_close_px", 3))
    seed_dil    = int(pcfg.get("seed_dilate_px", 4))
    compactness = float(pcfg.get("compactness", 0.0))
    use_mask    = bool(pcfg.get("use_map_mask", True))
    min_px      = int(pcfg.get("min_region_px", 64))
    points_file = pcfg.get("points_file", "Holnicote Apportionment Points.gpkg")
    mended_dir  = ROOT / pcfg.get("mended_dir", "data/mended outputs")
    mend_width  = int(pcfg.get("mended_line_width_px", 3))

    stitched_path = (Path(args.boundary) if args.boundary
                     else ROOT / paths["stitched"] / "boundaries" / f"{sheet}.tif")
    points_path   = resolve_points_file(ROOT / paths["parcel_points"], sheet, points_file)
    out_dir       = ROOT / paths["predictions"] / "parcels" / sheet
    out_geojson   = out_dir / "parcel_preds.geojson"
    out_preview   = out_dir / "parcel_segment_preview.png"

    if not points_path.exists():
        sys.exit(f"Apportionment points not found: {points_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sheet     : {sheet}")

    # ── Boundary source: prefer a hand-corrected GeoPackage if present ────────
    # The mended file holds the full corrected boundary polyline network (same
    # "boundaries" layer as data/outputs/<sheet>.gpkg).  We rasterise those lines
    # onto the sheet grid and use them instead of the raw model raster.
    mended_path = None if args.no_mended else resolve_mended(mended_dir, sheet)

    # Reference grid (transform / size / CRS) — from the stitched raster if it
    # exists, else the raw GeoTIFF.  Both share the same grid.
    grid_src = stitched_path if stitched_path.exists() else \
               (ROOT / paths["raw"] / sheet / f"{sheet}.tif")
    if not grid_src.exists():
        sys.exit(
            f"No boundary raster and no raw GeoTIFF to define the sheet grid:\n"
            f"  stitched: {stitched_path}\n  raw     : {grid_src}\n"
            "Run the lines pipeline (predict.py + vectorise.py) first."
        )
    with rasterio.open(grid_src) as src:
        transform = src.transform
        H, W = src.height, src.width
        try:
            epsg = src.crs.to_epsg() or 27700
        except Exception:
            epsg = 27700

    if mended_path is not None:
        print(f"Boundary  : {mended_path.relative_to(ROOT)}  (MENDED, rasterised @ {mend_width}px)")
        lines = read_gpkg_lines_wkb(mended_path)
        boundary = rasterize_lines(lines, transform, H, W, mend_width)
        print(f"  {len(lines):,} mended boundary lines  →  {W}×{H} px grid")
    else:
        if not stitched_path.exists():
            sys.exit(
                f"Boundary raster not found: {stitched_path}\n"
                "Run the lines pipeline first:\n"
                f"  python steps/04_predict/lines/predict.py --sheet {sheet}\n"
                f"  python steps/05_vectorise/lines/vectorise.py --sheet {sheet}"
            )
        with rasterio.open(stitched_path) as src:
            boundary = src.read(1)
        print(f"Boundary  : {stitched_path.relative_to(ROOT)}  (model raster)")
    print(f"Boundary px: {(boundary > 0).sum():,}  ({100*(boundary>0).mean():.2f}% of {W}×{H})")

    # ── Build the watershed surface (high = ridge/wall) ───────────────────────
    surf = (boundary > 0).astype(np.float32)
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px*2+1, close_px*2+1))
        surf = cv2.morphologyEx(surf, cv2.MORPH_CLOSE, k)
    if sigma > 0:
        surf = ndi.gaussian_filter(surf, sigma=sigma)
    if surf.max() > 0:
        surf = surf / surf.max()

    # ── Optional map-area mask ────────────────────────────────────────────────
    mask = None
    if use_mask:
        for mp in [ROOT / paths["masks"] / sheet / f"{sheet}.png",
                   ROOT / paths["masks"] / sheet / f"{sheet}.PNG",
                   ROOT / paths["masks"] / f"{sheet}.png"]:
            if mp.exists():
                raw = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if raw is not None:
                    if (raw.shape[1], raw.shape[0]) != (W, H):
                        raw = cv2.resize(raw, (W, H), interpolation=cv2.INTER_NEAREST)
                    mask = raw > 0
                    print(f"Map mask  : {mp.name}")
                    break
        if mask is None:
            print("Map mask  : none found — partitioning full sheet")

    # ── Seeds from apportionment points ───────────────────────────────────────
    print(f"Points file: {points_path.name}")
    pts = read_gpkg_points_wkb(points_path)
    inv_a = 1.0 / transform.a
    inv_e = 1.0 / transform.e
    markers = np.zeros((H, W), dtype=np.int32)
    label_to_rowid: dict[int, object] = {}
    label = 0
    seeded = skipped = 0
    for rec in pts:
        col = int(round((rec["_geom_x"] - transform.c) * inv_a))
        row = int(round((rec["_geom_y"] - transform.f) * inv_e))
        if not (0 <= col < W and 0 <= row < H):
            skipped += 1
            continue
        if mask is not None and not mask[row, col]:
            skipped += 1
            continue
        if markers[row, col] != 0:           # two points in same pixel — keep first
            continue
        label += 1
        markers[row, col] = label
        label_to_rowid[label] = rec.get("rowid", None)
        seeded += 1
    print(f"Points    : {seeded} seeded, {skipped} outside sheet/mask  (of {len(pts)})")
    if seeded == 0:
        sys.exit("No apportionment points fall within the sheet — nothing to segment.")

    if seed_dil > 0:
        # Grow each single-pixel seed into a small box so the marker is robust.
        # grey_dilation propagates the max label within the window — for seeds a
        # few px apart, collisions are vanishingly rare.  Crucially it is a
        # SEPARABLE box operation (two cheap 1-D passes, no large temporaries),
        # unlike skimage.expand_labels whose full-image distance transform with
        # return_indices allocates an ~8 GB int64 index array on big sheets and
        # gets OOM-killed (e.g. the 503M-px Luccombe sheet).  cv2.dilate is not an
        # option either — it rejects int32 label images.
        size = seed_dil * 2 + 1
        markers = ndi.grey_dilation(markers, size=(size, size)).astype(np.int32)

    # ── Watershed ─────────────────────────────────────────────────────────────
    print(f"Watershed : sigma={sigma} close={close_px} seed_dilate={seed_dil} "
          f"compactness={compactness} ...")
    labels = watershed(surf, markers=markers, mask=mask, compactness=compactness)

    # ── Vectorise the whole label raster as ONE coverage ──────────────────────
    # rasterio.features.shapes polygonises all labels in a single pass, tracing
    # along pixel GRID LINES.  Adjacent parcels therefore share *identical* edge
    # geometry (a gap-free, overlap-free coverage) — unlike per-label contour
    # tracing, which left ~1px slivers between neighbours.  Interior holes are
    # preserved too.  Coordinates come out in world units (transform applied).
    print("Vectorising coverage (rasterio.features.shapes) ...")
    feats = []
    dropped_small = 0
    counts = np.bincount(labels.ravel().astype(np.int64))   # px per label, one pass
    for geom, val in rio_shapes(labels.astype(np.int32), mask=(labels > 0),
                                transform=transform, connectivity=4):
        lab = int(val)
        if lab <= 0:
            continue
        px = int(counts[lab]) if lab < len(counts) else 0
        if px < min_px:
            dropped_small += 1
            continue
        rid = label_to_rowid.get(lab)
        feats.append({
            "type": "Feature",
            "geometry": geom,                  # GeoJSON dict, already world coords
            "properties": {
                "rowid": (int(rid) if rid is not None and rid == rid else None),
                "px_area": px,
            },
        })

    crs_member = {"type": "name",
                  "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}}
    doc = {"type": "FeatureCollection", "crs": crs_member, "features": feats}
    out_geojson.write_text(json.dumps(doc, separators=(",", ":")))
    features = feats   # for the summary print below

    print(f"\n{'─'*50}")
    print(f"Parcels written : {len(features)}  (dropped {dropped_small} < {min_px}px)")
    print(f"GeoJSON → {out_geojson.relative_to(ROOT)}")

    # ── Preview PNG (random colours per parcel over the boundary lines) ────────
    _write_preview(labels, boundary, out_preview)
    print(f"Preview → {out_preview.relative_to(ROOT)}")
    print(f"\nNext:  python steps/05_vectorise/parcels/parcel_vectorise.py --sheet {sheet}")


def _write_preview(labels: np.ndarray, boundary: np.ndarray, out_path: Path,
                   max_dim: int = 2500) -> None:
    """Colour each parcel randomly, overlay the boundary lines in black, downscale.

    Downsamples (by integer stride) BEFORE colourising so we never allocate a
    full-resolution RGB array — at ~400M px that would be >1 GB.
    """
    H, W = labels.shape
    step = max(1, int(np.ceil(max(H, W) / max_dim)))
    lab_s = labels[::step, ::step]
    bnd_s = boundary[::step, ::step]
    rng = np.random.default_rng(0)
    n = int(labels.max()) + 1
    lut = rng.integers(40, 255, size=(n, 3), dtype=np.uint8)
    lut[0] = (30, 30, 30)                        # background
    rgb = lut[lab_s.clip(0)]                      # small (h, w, 3)
    rgb[bnd_s > 0] = (0, 0, 0)                     # boundary lines on top
    cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
