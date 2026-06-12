"""
Prepare parcel segmentation training data from corrected QGIS predictions.

Workflow:
  1. Run parcel_predict.py + parcel_vectorise.py to generate initial predictions
  2. Open data/outputs/<sheet>.gpkg in QGIS
  3. In the 'parcels' layer: delete failed predictions, redraw incorrect boundaries
  4. Save the corrected layer back to the GeoPackage
  5. Run THIS script to extract image/mask pairs for fine-tuning

For each corrected parcel polygon this script:
  • Extracts a 1024×1024 px patch from the raw TIF centred on the polygon centroid
  • Rasterises the polygon boundary into a binary mask (0 / 255)
  • Saves the pair to data/annotations/parcels/<sheet>/images/ + masks/

Usage:
    conda activate polygons
    python steps/06_feedback/parcels/prepare_parcels.py --sheet Timberscombe
    python steps/06_feedback/parcels/prepare_parcels.py --sheet Timberscombe \\
        --min-area 50 --max-fill 0.7 --layer parcels
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

# ── PROJ / rasterio compat ────────────────────────────────────────────────────
if "PROJ_DATA" not in os.environ:
    _env_root = Path(sys.executable).parents[1]
    _cands = [_env_root / "share" / "proj"]
    import importlib.util as _ilu
    _s = _ilu.find_spec("pyproj")
    if _s and _s.submodule_search_locations:
        _p = Path(list(_s.submodule_search_locations)[0])
        _cands += [_p / "proj_dir" / "share" / "proj", _p / "data"]
    for _c in _cands:
        if (_c / "proj.db").exists():
            os.environ["PROJ_DATA"] = str(_c)
            break

try:
    import rasterio
    import rasterio.windows
except ImportError:
    sys.exit("rasterio required — conda activate polygons")


# ── GeoPackage helpers ────────────────────────────────────────────────────────

def _gpkg_flags(blob: bytes) -> tuple[int, int]:
    """Return (env_type, wkb_start) from a GeoPackage geometry blob header."""
    flags = blob[3]
    env_type  = (flags >> 1) & 0x07
    env_bytes = [0, 32, 48, 48, 64]
    return env_type, 8 + (env_bytes[env_type] if env_type < 5 else 0)


def parse_wkb_polygon(wkb: bytes) -> list[list[tuple[float, float]]]:
    """
    Parse a WKB Polygon (type 3) or PolygonZ (type 1003) and return a list of
    rings — each ring is a list of (X, Y) float tuples.  The exterior ring is
    always rings[0]; interior (hole) rings follow.
    Returns [] if the geometry is not a polygon.
    """
    endian   = "<" if wkb[0] == 1 else ">"
    raw_type = struct.unpack(endian + "I", wkb[1:5])[0]
    geom_t   = raw_type & 0xFFFF
    has_z    = geom_t in (1003, 3003) or bool(raw_type & 0x80000000)

    if geom_t not in (3, 1003, 3003):
        return []

    num_rings = struct.unpack(endian + "I", wkb[5:9])[0]
    off = 9
    rings: list[list[tuple[float, float]]] = []

    for _ in range(num_rings):
        num_pts = struct.unpack(endian + "I", wkb[off:off + 4])[0]
        off += 4
        stride = 24 if has_z else 16
        pts: list[tuple[float, float]] = []
        for _ in range(num_pts):
            x, y = struct.unpack(endian + "dd", wkb[off:off + 16])
            off  += stride
            pts.append((x, y))
        rings.append(pts)

    return rings


def read_polygons_from_gpkg(gpkg_path: Path, layer: str
                            ) -> list[dict]:
    """
    Read all features from a named GeoPackage layer using sqlite3.
    Returns list of dicts; geometry is decoded to rings via parse_wkb_polygon.
    Non-polygon features are silently skipped.
    """
    con = sqlite3.connect(str(gpkg_path))
    geom_col = con.execute(
        "SELECT column_name FROM gpkg_geometry_columns WHERE table_name=?",
        (layer,),
    ).fetchone()
    if geom_col is None:
        con.close()
        raise ValueError(f"Layer '{layer}' not found in {gpkg_path.name}")
    geom_col = geom_col[0]

    cur = con.execute(f"SELECT * FROM [{layer}]")
    col_names = [d[0] for d in cur.description]
    rows = cur.fetchall()
    con.close()

    records: list[dict] = []
    for row in rows:
        rd   = dict(zip(col_names, row))
        blob = rd.pop(geom_col, None)
        if not blob or len(blob) < 29:
            continue
        _, wkb_start = _gpkg_flags(blob)
        wkb   = blob[wkb_start:]
        rings = parse_wkb_polygon(wkb)
        if not rings:
            continue
        rd["_rings"] = rings
        records.append(rd)

    return records


# ── Geometry helpers ──────────────────────────────────────────────────────────

def polygon_centroid(exterior: list[tuple[float, float]]
                     ) -> tuple[float, float]:
    """Signed-area centroid of the exterior ring (Shoelace formula)."""
    n   = len(exterior)
    area = cx = cy = 0.0
    for i in range(n):
        xi, yi  = exterior[i]
        xi1, yi1 = exterior[(i + 1) % n]
        cross    = xi * yi1 - xi1 * yi
        area    += cross
        cx      += (xi + xi1) * cross
        cy      += (yi + yi1) * cross
    area /= 2.0
    if abs(area) < 1e-10:
        # Degenerate — fall back to mean
        xs, ys = zip(*exterior)
        return float(np.mean(xs)), float(np.mean(ys))
    cx /= (6.0 * area)
    cy /= (6.0 * area)
    return cx, cy


def rings_to_pixel_coords(rings: list[list[tuple[float, float]]],
                           tif_tf, patch_col: int, patch_row: int,
                           patch_size: int) -> list[np.ndarray]:
    """
    Convert BNG ring coordinates → pixel coords within the extracted patch.
    Returns list of (N, 2) int32 arrays (one per ring), clipped to patch bounds.
    """
    px_rings = []
    for ring in rings:
        cols = [(x - tif_tf.c) / tif_tf.a - patch_col for x, _ in ring]
        rows = [(y - tif_tf.f) / tif_tf.e - patch_row for _, y in ring]
        pts  = np.column_stack([cols, rows]).astype(np.float32)
        pts  = np.clip(pts, 0, patch_size - 1).astype(np.int32)
        px_rings.append(pts.reshape(-1, 1, 2))
    return px_rings


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract image/mask training pairs from corrected parcels."
    )
    parser.add_argument("--sheet",    required=True)
    parser.add_argument("--layer",    default="parcels",
                        help="GeoPackage layer name (default: parcels)")
    parser.add_argument("--min-area", type=float, default=50.0,
                        help="Skip polygons smaller than this area in m² (default 50)")
    parser.add_argument("--max-fill", type=float, default=0.70,
                        help="Skip polygons whose mask fills > this fraction of patch "
                             "(likely unfixed bad predictions — default 0.70)")
    args = parser.parse_args()
    sheet_id = args.sheet

    cfg   = yaml.safe_load((ROOT / "config.yaml").read_text())
    paths = cfg["paths"]
    pcfg  = cfg.get("parcels", {})

    patch_size = int(pcfg.get("tile_size", 1024))

    gpkg_path = ROOT / paths["outputs"]      / f"{sheet_id}.gpkg"
    tif_path  = ROOT / paths["raw"]          / sheet_id / f"{sheet_id}.tif"
    out_dir   = ROOT / paths["annotations"]  / "parcels" / sheet_id

    if not gpkg_path.exists():
        sys.exit(f"GeoPackage not found: {gpkg_path}\n"
                 "Run parcel_predict.py + parcel_vectorise.py + QGIS correction first.")
    if not tif_path.exists():
        sys.exit(f"TIF not found: {tif_path}")

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    # ── Read polygons ─────────────────────────────────────────────────────────
    print(f"Reading '{args.layer}' layer from {gpkg_path.name} ...")
    records = read_polygons_from_gpkg(gpkg_path, args.layer)
    print(f"  {len(records)} polygon features found")

    # ── TIF metadata ──────────────────────────────────────────────────────────
    with rasterio.open(tif_path) as src:
        tif_tf = src.transform
        tif_w  = src.width
        tif_h  = src.height
        n_bands = src.count

    half = patch_size // 2
    saved = skipped = 0
    meta: list[dict] = []

    with rasterio.open(tif_path) as src:
        for rec in records:
            exterior = rec["_rings"][0]

            # ── Polygon area filter ───────────────────────────────────────────
            # Quick pixel-space area via Shoelace on exterior ring
            cx_bng, cy_bng = polygon_centroid(exterior)
            px_all = [(x - tif_tf.c) / tif_tf.a for x, _ in exterior]
            py_all = [(y - tif_tf.f) / tif_tf.e for _, y in exterior]
            n_ext  = len(exterior)
            area_px = abs(sum(
                (px_all[i] * py_all[(i+1)%n_ext] - px_all[(i+1)%n_ext] * py_all[i])
                for i in range(n_ext)
            )) / 2.0
            area_m2 = area_px * (abs(tif_tf.a) ** 2)

            if area_m2 < args.min_area:
                skipped += 1
                continue

            # ── Compute patch origin ──────────────────────────────────────────
            cx_px = (cx_bng - tif_tf.c) / tif_tf.a
            cy_px = (cy_bng - tif_tf.f) / tif_tf.e

            patch_col = int(np.clip(round(cx_px - half), 0, tif_w - patch_size))
            patch_row = int(np.clip(round(cy_px - half), 0, tif_h - patch_size))

            # ── Read image patch ──────────────────────────────────────────────
            win   = rasterio.windows.Window(patch_col, patch_row, patch_size, patch_size)
            patch = np.full((patch_size, patch_size, 3), 255, dtype=np.uint8)

            actual_w = min(patch_size, tif_w - patch_col)
            actual_h = min(patch_size, tif_h - patch_row)
            if actual_w > 0 and actual_h > 0:
                win = rasterio.windows.Window(patch_col, patch_row, actual_w, actual_h)
                if n_bands >= 3:
                    data = src.read([1, 2, 3], window=win)
                else:
                    grey = src.read(1, window=win)
                    data = np.stack([grey, grey, grey], axis=0)
                patch[:actual_h, :actual_w] = np.transpose(data, (1, 2, 0))

            # ── Rasterise mask ────────────────────────────────────────────────
            mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
            px_rings = rings_to_pixel_coords(rec["_rings"], tif_tf,
                                             patch_col, patch_row, patch_size)
            # Fill exterior ring, then subtract holes (rings[1:])
            cv2.fillPoly(mask, [px_rings[0]], 255)
            for hole in px_rings[1:]:
                cv2.fillPoly(mask, [hole], 0)

            # ── Fill-fraction filter ──────────────────────────────────────────
            fill = float((mask > 0).mean())
            if fill > args.max_fill:
                skipped += 1
                continue
            if fill < 1e-4:          # empty mask — probably outside patch bounds
                skipped += 1
                continue

            # ── Save ──────────────────────────────────────────────────────────
            rowid = rec.get("rowid", saved)
            name  = f"parcel_{rowid:06}"
            cv2.imwrite(str(out_dir / "images" / f"{name}.png"), patch)
            cv2.imwrite(str(out_dir / "masks"  / f"{name}.png"), mask)

            meta.append({
                "name": name, "rowid": rowid,
                "patch_col": patch_col, "patch_row": patch_row,
                "cx_px": round(cx_px, 2), "cy_px": round(cy_px, 2),
                "area_m2": round(area_m2, 1), "fill": round(fill, 4),
            })
            saved += 1

    # Save metadata (used by train.py for centroid-based prompt simulation)
    meta_path = out_dir / "parcels.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nSaved  : {saved} pairs → {out_dir.relative_to(ROOT)}")
    print(f"Skipped: {skipped} (area < {args.min_area} m² or fill > {args.max_fill})")
    print(f"Meta   : {meta_path.name}")
    print(f"\nNext:  python steps/06_feedback/parcels/train.py --sheet {sheet_id}")


if __name__ == "__main__":
    main()
