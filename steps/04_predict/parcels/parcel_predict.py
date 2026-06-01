"""
Point-prompted SAM parcel segmentation.

Tiles the raw sheet GeoTIFF at tile_size × tile_size px (default 2048), downsamples to
sam_input_size (default 1024) for SAM ViT-B inference, uses apportionment centroid points
from the parcel_points GeoPackage as SAM prompts, and writes predicted parcel polygons to:

    data/predictions/parcels/<SHEET_ID>/parcel_preds.geojson

No model training or download needed — uses the existing SAM ViT-B weights at:
    models/base/MapSAM/origional_weights/sam_vit_b_01ec64.pth

Usage:
    conda activate polygons
    python steps/04_predict/parcels/parcel_predict.py --sheet Timberscombe
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT       = Path(__file__).resolve().parents[3]
MAPSAM_DIR = ROOT / "models" / "MapSAM"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MAPSAM_DIR))

# ── PROJ database fix ─────────────────────────────────────────────────────────
# pyproj sometimes cannot locate its PROJ database inside conda environments,
# especially when pyproj was pip-installed (database lives inside the package)
# rather than conda-installed (database lives in share/proj).
# Search candidate locations and set PROJ_DATA to the first that contains proj.db.
if "PROJ_DATA" not in os.environ:
    _env_root = Path(sys.executable).parents[1]
    _proj_candidates: list[Path] = [
        _env_root / "share" / "proj",                              # conda-forge pyproj
    ]
    # pip-installed pyproj bundles proj.db inside the package directory
    import importlib.util as _ilu
    _spec = _ilu.find_spec("pyproj")
    if _spec and _spec.submodule_search_locations:
        _pkg = Path(list(_spec.submodule_search_locations)[0])
        _proj_candidates += [
            _pkg / "proj_dir" / "share" / "proj",                  # pip pyproj ≥ 3.x
            _pkg / "data",                                          # older pip pyproj
        ]
    for _p in _proj_candidates:
        if (_p / "proj.db").exists():
            os.environ["PROJ_DATA"] = str(_p)
            break

# ── pyogrio sanity-check ──────────────────────────────────────────────────────
# pyogrio can be broken in pip-installed or partially-uninstalled states:
#   • PROJ version mismatch  → crashes on import with ValueError
#   • Missing binary extension after pip uninstall → imports but lacks read_dataframe
# In either case, evict it from sys.modules so geopandas falls back to fiona.
_pyogrio_ok = False
try:
    import pyogrio as _pyogrio_test
    _ = _pyogrio_test.read_dataframe   # confirm the binary extension is present
    _pyogrio_ok = True
except ValueError as _e:
    if "DATABASE.LAYOUT.VERSION" in str(_e) or "proj_data" in str(_e).lower():
        print(
            "Warning: pyogrio has a PROJ database version conflict — "
            "falling back to fiona.\n"
            "To clean up permanently: pip uninstall pyogrio"
        )
    # fall through to eviction below
except (ImportError, AttributeError):
    pass  # broken / partial install — evict silently

if not _pyogrio_ok:
    # Block pyogrio so geopandas cannot find it and is forced onto fiona.
    # sys.modules[name] = None is the documented way to make 'import name'
    # raise ImportError in all subsequent code.
    for _k in list(sys.modules.keys()):
        if _k == "pyogrio" or _k.startswith("pyogrio."):
            del sys.modules[_k]
    sys.modules["pyogrio"] = None  # type: ignore[assignment]

try:
    import rasterio
    import rasterio.windows
    from rasterio.transform import Affine
except ImportError:
    sys.exit(
        "rasterio is required but not installed.\n"
        "Run once:  conda activate polygons && conda install -c conda-forge rasterio geopandas"
    )

import pandas as pd                                                # noqa: E402
import sqlite3                                                       # noqa: E402
import struct                                                        # noqa: E402
from shapely.geometry import mapping                                  # noqa: E402
# Note: geopandas is intentionally NOT imported here.  pyproj (its CRS backend)
# may be broken in the polygons conda environment.  Point coordinates are read
# directly from the GeoPackage via sqlite3 + WKB decoding (no CRS resolution
# needed since both TIF and GeoPackage are in EPSG:27700).

import torch                                                       # noqa: E402
from segment_anything import SamPredictor, sam_model_registry     # noqa: E402
from tqdm import tqdm                                              # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config() -> dict:
    p = ROOT / "config.yaml"
    if not p.exists():
        sys.exit(f"config.yaml not found: {p}")
    return yaml.safe_load(p.read_text())


def read_tile_rgb(src: "rasterio.DatasetReader", col: int, row: int,
                  tile_size: int) -> np.ndarray:
    """
    Read a tile_size × tile_size pixel window starting at (col, row) from src.
    Pads right / bottom with 255 (white) if the window extends past the raster edge.
    Returns a C-contiguous (H, W, 3) uint8 RGB array suitable for SAM.
    Handles both 1-band (grayscale) and 3-band (RGB) TIFs.
    """
    actual_w = max(0, min(tile_size, src.width  - col))
    actual_h = max(0, min(tile_size, src.height - row))

    tile = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)

    if actual_w > 0 and actual_h > 0:
        win = rasterio.windows.Window(col, row, actual_w, actual_h)
        if src.count >= 3:
            data = src.read([1, 2, 3], window=win)      # (3, H, W)
            tile[:actual_h, :actual_w] = np.transpose(data, (1, 2, 0))
        else:
            # Grayscale TIF — read band 1 and replicate to RGB
            grey = src.read(1, window=win)               # (H, W)
            rgb  = np.stack([grey, grey, grey], axis=-1) # (H, W, 3)
            tile[:actual_h, :actual_w] = rgb

    return np.ascontiguousarray(tile)


def read_gpkg_points_wkb(path: Path) -> pd.DataFrame:
    """
    Read point features from a GeoPackage using sqlite3 + WKB decoding.
    Returns a DataFrame with all attribute columns plus _geom_x and _geom_y.
    No geopandas or pyproj required — avoids any PROJ database issues.
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
        row_dict   = dict(zip(col_names, row))
        geom_blob  = row_dict.pop(geom_col, None)
        if not geom_blob or len(geom_blob) < 29:   # 8 header + 21 WKB Point minimum
            continue

        # ── GeoPackage geometry header (GPKG spec §2.1.3) ────────────────────
        # byte 0-1 : magic 0x47 0x50 ('GP')
        # byte 2   : version
        # byte 3   : flags — bits[1:3] = envelope type, bit[0] = empty flag
        # byte 4-7 : SRS ID (int32, ignored here)
        # byte 8+  : optional envelope, then standard WKB
        flags = geom_blob[3]
        # Note: do NOT trust the empty-geometry flag (bit 0).  Some tools (e.g. QGIS
        # exports) incorrectly set this bit even when the geometry blob is valid.
        # We check for an actually-short blob instead.
        env_type  = (flags >> 1) & 0x07
        env_bytes = [0, 32, 48, 48, 64]          # bytes per envelope type 0-4
        wkb_start = 8 + (env_bytes[env_type] if env_type < 5 else 0)
        wkb       = geom_blob[wkb_start:]

        # ── WKB Point ────────────────────────────────────────────────────────
        # byte 0   : byte order (1=little-endian, 0=big-endian)
        # bytes 1-4: geometry type uint32
        # bytes 5-12: X (float64)
        # bytes 13-20: Y (float64)  [Z at 21-29 is ignored for PointZ]
        #
        # WKB geometry type variants for points:
        #   1    = Point       (OGC/simple WKB)
        #   1001 = PointZ      (ISO WKB)
        #   2001 = PointM      (ISO WKB)
        #   3001 = PointZM     (ISO WKB)
        #   0x80000001 → & 0xFFFF = 1   (EWKB PointZ, high bit = has-Z flag)
        # X and Y are always at bytes 5-21 regardless of Z/M presence.
        endian    = "<" if wkb[0] == 1 else ">"
        raw_type  = struct.unpack(endian + "I", wkb[1:5])[0]
        geom_type = raw_type & 0xFFFF   # strip EWKB high-bit flags
        # Accept any Point variant (1, 1001, 2001, 3001)
        if geom_type not in (1, 1001, 2001, 3001):
            continue
        x, y = struct.unpack(endian + "dd", wkb[5:21])
        row_dict["_geom_x"] = x
        row_dict["_geom_y"] = y
        records.append(row_dict)

    return pd.DataFrame(records)


def set_image_safe(predictor: "SamPredictor", image_rgb: np.ndarray) -> None:
    """
    Drop-in replacement for predictor.set_image() compatible with NumPy 2.x.

    Both torch.as_tensor() and torch.from_numpy() check for the C-level
    numpy.ndarray type, which changed in NumPy 2.0 making them fail with
    "expected np.ndarray (got numpy.ndarray)" when PyTorch was compiled
    against NumPy 1.x.

    The fix: convert through memoryview (pure Python buffer protocol) which
    PyTorch accepts via torch.frombuffer() regardless of NumPy version.
    We also replace the torchvision resize with cv2 to keep the same path.
    """
    h, w = image_rgb.shape[:2]
    th, tw = predictor.transform.get_preprocess_shape(
        h, w, predictor.transform.target_length
    )
    resized = cv2.resize(image_rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
    arr = np.ascontiguousarray(resized, dtype=np.uint8)   # (H, W, 3) C-contiguous

    # memoryview exposes the buffer protocol without touching numpy's type system
    # .clone() detaches from the transient buffer so the tensor owns its storage
    t = (torch.frombuffer(memoryview(arr), dtype=torch.uint8)
              .reshape(arr.shape)
              .clone()
              .to(predictor.device)
              .permute(2, 0, 1)
              .contiguous()[None])          # → (1, 3, H, W)

    predictor.set_torch_image(t, (h, w))

    # MapSAM's image encoder returns (embedding_tensor, intermediate_states) — a tuple.
    # The standard predictor stores it as-is, then the mask decoder calls
    # repeat_interleave(image_embeddings, ...) and fails because it's a tuple not a tensor.
    # Unwrap here so predictor.features is always the plain embedding tensor.
    if isinstance(predictor.features, tuple):
        predictor.features = predictor.features[0]


def vectorise_mask(mask: np.ndarray, transform: "Affine") -> "shapely_shape | None":
    """
    Convert a boolean or uint8 (0/1) SAM mask to a single Shapely polygon in the
    coordinate system defined by `transform`.  Returns the largest connected region,
    or None if the mask is empty.

    Uses cv2.findContours instead of rasterio.features.shapes to avoid
    the NumPy 2.0 / rasterio dtype-resolution incompatibility
    (rasterio.dtypes._getnpdtype calls numpy.dtype(x) which fails in NumPy 2.0
    when rasterio was compiled against NumPy 1.x).

    Coordinate conversion:
        col_px, row_px  →  X_crs = transform.c + col_px * transform.a
                            Y_crs = transform.f + row_px * transform.e
    (Affine off-diagonal terms are zero for axis-aligned rasters.)
    """
    from shapely.geometry import Polygon

    # np.array(..., dtype=np.uint8) creates a fresh NumPy-2.0-native array,
    # avoiding any stale internal dtype from a torch .numpy() call.
    mask_u8 = np.array(mask, dtype=np.uint8)
    if not mask_u8.any():
        return None

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Pick the largest exterior contour by area
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 1:
        return None

    # contour shape is (N, 1, 2) → squeeze to (N, 2) float64 (col=x, row=y)
    pts_px = largest.reshape(-1, 2).astype(np.float64)

    # Apply affine transform: (col, row) → (Easting, Northing)
    pts_crs = np.column_stack([
        transform.c + pts_px[:, 0] * transform.a,   # X = origin_X + col * pixel_w
        transform.f + pts_px[:, 1] * transform.e,   # Y = origin_Y + row * pixel_h  (negative)
    ])

    if len(pts_crs) < 3:
        return None

    poly = Polygon(pts_crs)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if not poly.is_empty else None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Point-prompted SAM parcel segmentation."
    )
    parser.add_argument("--sheet",     required=True,
                        help="Sheet ID — must match a subfolder in data/raw/")
    parser.add_argument("--device",    default=None,
                        help="'cuda', 'cpu', or omit for auto-detect")
    parser.add_argument("--tile-size", type=int, default=None,
                        help="Override tile_size from config.yaml")
    parser.add_argument("--overlap",   type=int, default=None,
                        help="Override overlap from config.yaml")
    args = parser.parse_args()
    sheet_id = args.sheet

    cfg   = load_config()
    paths = cfg["paths"]
    pcfg  = cfg.get("parcels", {})

    tile_size   = args.tile_size or int(pcfg.get("tile_size",     2048))
    sam_size    = int(pcfg.get("sam_input_size", 1024))
    overlap     = args.overlap   or int(pcfg.get("overlap",        256))
    points_file = pcfg.get("points_file", "Holnicote Apportionment Points.gpkg")

    stride      = tile_size - overlap
    scale       = sam_size / tile_size   # 0.5 for 2048 → 1024
    half_ov     = overlap // 2

    tif_path    = ROOT / paths["raw"]          / sheet_id / f"{sheet_id}.tif"
    points_path = ROOT / paths["parcel_points"] / points_file
    out_dir     = ROOT / paths["predictions"]  / "parcels" / sheet_id
    out_geojson = out_dir / "parcel_preds.geojson"

    for p, label in [(tif_path, "TIF"), (points_path, "parcel points GeoPackage")]:
        if not p.exists():
            sys.exit(f"{label} not found: {p}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device    : {device}")

    # ── Load SAM ──────────────────────────────────────────────────────────────
    # num_classes=4 → standard SAM architecture (3 multi-mask outputs + 1 single)
    sam_ckpt = (ROOT / paths["models_base"]
                / "MapSAM" / "origional_weights" / "sam_vit_b_01ec64.pth")
    if not sam_ckpt.exists():
        sys.exit(f"SAM checkpoint not found: {sam_ckpt}")

    sam, _ = sam_model_registry["vit_b"](
        image_size  = sam_size,
        num_classes = 4,
        checkpoint  = str(sam_ckpt),
    )
    sam.eval().to(device)
    predictor = SamPredictor(sam)
    print(f"SAM ViT-B : {sam_ckpt.name}\n")

    # ── TIF metadata ──────────────────────────────────────────────────────────
    with rasterio.open(tif_path) as src:
        tif_tf     = src.transform   # Affine: pixel (col, row) → CRS (X, Y)
        tif_w      = src.width
        tif_h      = src.height
        tif_bounds = src.bounds      # BoundingBox(left, bottom, right, top)
        # src.crs uses pyproj which may be broken; we know these maps are EPSG:27700
        try:
            epsg_out = src.crs.to_epsg() or 27700
        except Exception:
            epsg_out = 27700

    # ── Parcel points (sqlite3 + WKB — no pyproj required) ────────────────────
    print("Reading parcel points ...")
    all_pts = read_gpkg_points_wkb(points_path)
    if all_pts.empty or "_geom_x" not in all_pts.columns:
        sys.exit(
            f"No point geometries could be decoded from {points_path.name}.\n"
            "This usually means the WKB geometry type is unrecognised.\n"
            "Run with --debug-gpkg to inspect the raw bytes."
        )
    print(f"  Loaded {len(all_pts)} points from GeoPackage")

    # Spatial filter: plain coordinate comparison (TIF and GeoPackage are both EPSG:27700)
    pts_in = all_pts[
        (all_pts["_geom_x"] >= tif_bounds.left)   &
        (all_pts["_geom_x"] <= tif_bounds.right)  &
        (all_pts["_geom_y"] >= tif_bounds.bottom) &
        (all_pts["_geom_y"] <= tif_bounds.top)
    ].copy().reset_index(drop=True)

    print(f"Sheet     : {sheet_id}  ({tif_w}×{tif_h} px, {abs(tif_tf.a):.2f} m/px)")
    print(f"Points    : {len(pts_in)} within sheet  (of {len(all_pts)} total)")
    if len(pts_in) == 0:
        sys.exit("No parcel points found within sheet bounds.")

    # Pixel-space columns (float, in TIF pixel coordinates)
    pts_in["_px_col"] = (pts_in["_geom_x"] - tif_tf.c) / tif_tf.a
    pts_in["_px_row"] = (pts_in["_geom_y"] - tif_tf.f) / tif_tf.e

    # ── Tile grid ─────────────────────────────────────────────────────────────
    col_cores = list(range(0, tif_w, stride))
    row_cores = list(range(0, tif_h, stride))
    n_tiles   = len(col_cores) * len(row_cores)

    print(f"Tile grid : {len(col_cores)}×{len(row_cores)} = {n_tiles} tiles "
          f"(size={tile_size}, overlap={overlap}, stride={stride})")
    print(f"Scale     : {tile_size}→{sam_size} px  (×{scale:.2f})\n")

    features_out: list[dict] = []
    skipped = 0
    empty   = 0

    tile_list = [(ci, ri, cs, rs)
                 for ri, rs in enumerate(row_cores)
                 for ci, cs in enumerate(col_cores)]

    with rasterio.open(tif_path) as src:
        for ci, ri, col_core, row_core in tqdm(tile_list, desc="Tiles", unit="tile"):

            # ── Core region (home for points assigned to this tile) ──────────
            col_core_end = min(col_core + stride, tif_w)
            row_core_end = min(row_core + stride, tif_h)

            # ── Read region (offset half-overlap left/up for context) ─────────
            col_read = max(0, col_core - half_ov)
            row_read = max(0, row_core - half_ov)

            # ── Points whose home is this tile's core ─────────────────────────
            tile_pts = pts_in[
                (pts_in["_px_col"] >= col_core) & (pts_in["_px_col"] < col_core_end) &
                (pts_in["_px_row"] >= row_core) & (pts_in["_px_row"] < row_core_end)
            ]
            if len(tile_pts) == 0:
                continue

            # ── Read + downsample tile ────────────────────────────────────────
            tile_rgb = read_tile_rgb(src, col_read, row_read, tile_size)
            tile_sam = np.ascontiguousarray(
                cv2.resize(tile_rgb, (sam_size, sam_size),
                           interpolation=cv2.INTER_AREA),
                dtype=np.uint8,
            )

            set_image_safe(predictor, tile_sam)   # encodes once per tile

            # Affine transform from SAM pixel → BNG (for vectorisation)
            # Pixel size at SAM scale = original_pixel_size / scale
            sam_tf = Affine(
                tif_tf.a / scale, 0.0, tif_tf.c + col_read * tif_tf.a,
                0.0, tif_tf.e / scale, tif_tf.f + row_read * tif_tf.e,
            )

            # ── Per-point SAM inference ───────────────────────────────────────
            for _, pt in tile_pts.iterrows():
                # Point in SAM pixel coordinates (X = col, Y = row)
                sam_px_x = float((pt["_px_col"] - col_read) * scale)
                sam_px_y = float((pt["_px_row"] - row_read) * scale)

                try:
                    masks, scores, _ = predictor.predict(
                        point_coords   = np.array([[sam_px_x, sam_px_y]]),
                        point_labels   = np.array([1]),   # 1 = foreground
                        multimask_output = True,
                    )
                except Exception as exc:
                    tqdm.write(f"  SAM failed (rowid={pt.get('rowid','?')}): {exc}")
                    skipped += 1
                    continue

                best_idx = int(np.argmax(scores))
                poly = vectorise_mask(masks[best_idx], sam_tf)

                if poly is None:
                    empty += 1
                    continue

                rowid_val = pt.get("rowid", None)
                features_out.append({
                    "type"      : "Feature",
                    "geometry"  : mapping(poly),
                    "properties": {
                        "rowid"     : (int(rowid_val)
                                       if rowid_val is not None and rowid_val == rowid_val
                                       else None),
                        "sam_score" : float(scores[best_idx]),
                        "tile_col"  : ci,
                        "tile_row"  : ri,
                    },
                })

    print(f"\n{'─'*50}")
    print(f"Polygons written : {len(features_out)}")
    print(f"Empty masks      : {empty}")
    print(f"SAM errors       : {skipped}")

    crs_member = ({"type": "name",
                   "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg_out}"}}
                  if epsg_out else None)

    geojson_doc = {
        "type"    : "FeatureCollection",
        "features": features_out,
    }
    if crs_member:
        geojson_doc["crs"] = crs_member

    out_geojson.write_text(json.dumps(geojson_doc, separators=(",", ":")))
    print(f"Written → {out_geojson.relative_to(ROOT)}")
    print(f"\nNext step:  python steps/05_vectorise/parcels/parcel_vectorise.py"
          f" --sheet {sheet_id}")


if __name__ == "__main__":
    main()
