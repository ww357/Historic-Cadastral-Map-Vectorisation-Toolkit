"""
Reproject a georeferenced map to British National Grid (or any target CRS) as a
tiled GeoTIFF, so the rest of the pipeline runs in metres at the training scale.

Why this step exists
--------------------
Maps arrive in whatever CRS they were georeferenced in — often WGS84 degrees from
GCP warping.  The toolkit's thresholds (simplify_tolerance, min_length, min_area,
the parcel size cap) are all in metres, and the models were trained on ~0.5 m/px
BNG scans, so a degree-CRS map at a different pixel scale predicts poorly and its
metric thresholds are meaningless.  Reprojecting once at ingest fixes all of that,
and as a bonus:
  * warps to a north-up, axis-aligned raster (removes rotated-transform edge cases)
  * writes a TILED GeoTIFF, so patchify's windowed reads are fast (no JPEG re-decode)
  * resamples to the training resolution so features match what the models expect

Output is written to data/raw/<SHEET>/<SHEET>.tif.  Because find_raw() prefers
.tif over .vrt/.jpg/.png, patchify then picks the reprojected map automatically;
the original source files stay in place as provenance.

Ungeoreferenced inputs (a plain JPG/PNG with no CRS) cannot be reprojected — there
is nothing to reproject FROM — so they are left as-is and patchify handles them in
pixel coordinates.

Config (config.yaml, `reproject:`):
    target_crs   EPSG code / WKT               (default EPSG:27700, British National Grid)
    resolution   target ground resolution      (default 0.5, metres per pixel)
    resampling   nearest|bilinear|cubic|lanczos|average  (default bilinear)

Usage:
    conda activate maptools
    python steps/01_patchify/reproject.py --sheet SHEET_ID
    python steps/01_patchify/reproject.py --sheet SHEET_ID --resolution 0.25 --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rasterio
import yaml
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject

ROOT = Path(__file__).resolve().parents[2]

# Raw map formats, in resolution priority (matches patchify.py).
RAW_EXTENSIONS = (".tif", ".tiff", ".vrt", ".jpg", ".jpeg", ".png")

_RESAMPLING = {
    "nearest":  Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic":    Resampling.cubic,
    "lanczos":  Resampling.lanczos,
    "average":  Resampling.average,
}


def load_config() -> dict:
    p = ROOT / "config.yaml"
    if not p.exists():
        sys.exit(f"config.yaml not found at {p}")
    return yaml.safe_load(p.read_text())


def find_raw(raw_root: Path, sheet_id: str) -> Path | None:
    for ext in RAW_EXTENSIONS:
        p = raw_root / sheet_id / f"{sheet_id}{ext}"
        if p.exists():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproject a map to BNG (tiled GeoTIFF).")
    ap.add_argument("--sheet", required=True, help="Sheet ID")
    ap.add_argument("--target-crs", default=None, help="Override target CRS (e.g. EPSG:27700)")
    ap.add_argument("--resolution", type=float, default=None,
                    help="Override target ground resolution (metres per pixel)")
    ap.add_argument("--resampling", default=None, choices=list(_RESAMPLING),
                    help="Override resampling method")
    ap.add_argument("--force", action="store_true",
                    help="Reproject even if the source is already in the target CRS.")
    args = ap.parse_args()
    sheet = args.sheet

    cfg  = load_config()
    rcfg = cfg.get("reproject", {})
    target_crs = args.target_crs or rcfg.get("target_crs", "EPSG:27700")
    resolution = args.resolution if args.resolution is not None else float(rcfg.get("resolution", 0.5))
    method     = args.resampling or rcfg.get("resampling", "bilinear")
    resampling = _RESAMPLING[method]

    raw_root = ROOT / cfg["paths"]["raw"]
    src_path = find_raw(raw_root, sheet)
    if src_path is None:
        exts = "/".join(e.lstrip(".") for e in RAW_EXTENSIONS)
        sys.exit(f"Raw map not found: {raw_root / sheet / sheet}.({exts})")
    out_path = raw_root / sheet / f"{sheet}.tif"

    try:
        target_epsg = CRS.from_string(target_crs).to_epsg()
    except Exception:
        target_epsg = None

    with rasterio.open(src_path) as src:
        print(f"Sheet   : {sheet}")
        print(f"Source  : {src_path.name}  |  {src.width}x{src.height}px  |  {src.driver}")

        if src.crs is None:
            sys.exit(
                f"'{src_path.name}' is ungeoreferenced (no CRS) — nothing to reproject.\n"
                "Patchify will process it in pixel coordinates as-is."
            )

        src_epsg = src.crs.to_epsg()
        print(f"Source CRS : {src.crs.to_string()}  (EPSG:{src_epsg})")
        print(f"Target CRS : {target_crs}  |  {resolution} m/px  |  {method}")

        # Already in the target CRS as a GeoTIFF — leave it unless forced.
        if src_epsg is not None and src_epsg == target_epsg and src.driver == "GTiff" and not args.force:
            print(f"Already {target_crs} GeoTIFF — nothing to do (use --force to re-tile/resample).")
            return

        # Refuse to overwrite the source in place (e.g. a non-BNG <sheet>.tif) so the
        # original is never lost; ask the user to rename it first.
        if src_path.resolve() == out_path.resolve():
            sys.exit(
                f"Source is already {out_path.name}; reprojecting would overwrite it.\n"
                f"Rename the original (e.g. {sheet}_source.tif) and re-run."
            )

        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds, resolution=resolution,
        )

        profile = src.profile.copy()
        profile.update(
            driver="GTiff", crs=target_crs, transform=transform,
            width=width, height=height,
            tiled=True, blockxsize=512, blockysize=512, compress="LZW",
        )

        print(f"Output  : {out_path.relative_to(ROOT)}  |  {width}x{height}px  (tiled GeoTIFF)")
        with rasterio.open(out_path, "w", **profile) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=transform, dst_crs=target_crs,
                    resampling=resampling,
                )

    print(
        f"\nDone. find_raw now prefers {out_path.name}, so the rest of the pipeline uses it.\n"
        f"Next step:\n"
        f"  python steps/01_patchify/draw_mask.py --sheet {sheet}   (optional map-area mask)\n"
        f"  python steps/01_patchify/patchify.py --sheet {sheet} [--mask]"
    )


if __name__ == "__main__":
    main()
