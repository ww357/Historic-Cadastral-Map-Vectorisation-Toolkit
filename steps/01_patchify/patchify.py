"""
Slice a map sheet into 512px PNG patches.

The raw map may be any of GeoTIFF (.tif/.tiff), GDAL VRT (.vrt), or JPG/PNG,
georeferenced or not — data/raw/<SHEET_ID>/<SHEET_ID>.<ext>, tried in that
priority order.  Georeferencing is read straight from the file: embedded for
GeoTIFF/VRT, or from sidecar world file + projection (e.g. <SHEET>.jgw + .prj)
which GDAL picks up automatically.  A plain JPG/PNG with no sidecars is treated
as ungeoreferenced and processed in pixel coordinates (has_georef=False),
exactly like a non-georef GeoTIFF.

JPG/PNG cannot be windowed efficiently (no internal tiling), so those formats
are read fully into memory once and sliced from the array; GeoTIFF/VRT keep
memory-safe per-patch windowed reads.

A map-area mask is applied automatically if one is found in
data/map_area_masks/<SHEET_ID>/ (.png, .tif, or .tiff). Only patches that
overlap the mask by at least min_mask_coverage (config.yaml) are then saved.

Use --mask to require a mask and exit with an error if none is found.
Omit --mask to apply the mask when present and proceed without it when absent.

All spatial metadata needed for stitching predictions back into a full-document
raster is written to a CSV alongside the patches.

Usage:
    python patchify.py --sheet SHEET_ID [--mask]

Outputs:
    data/patches/images/<SHEET_ID>/<patch_id>.png
    data/patches/metadata/<SHEET_ID>_patches.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from PIL import Image
from rasterio.transform import Affine
from rasterio.windows import Window
from tqdm import tqdm


def load_config(repo_root: Path) -> dict:
    path = repo_root / "config.yaml"
    if not path.exists():
        sys.exit(f"config.yaml not found at {path}")
    return yaml.safe_load(path.read_text())


def patch_grid(img_w: int, img_h: int, size: int, overlap: int) -> list[tuple]:
    step = size - overlap
    return [(c, r) for r in range(0, img_h, step) for c in range(0, img_w, step)]


def patch_transform(base: Affine, col_off: int, row_off: int) -> Affine:
    return Affine(
        base.a, base.b, base.c + col_off * base.a,
        base.d, base.e, base.f + row_off * base.e,
    )


def load_mask(mask_path: Path) -> np.ndarray:
    """Return a 2-D boolean array (True = inside map area)."""
    if mask_path.suffix.lower() == ".png":
        Image.MAX_IMAGE_PIXELS = None   # large map masks exceed PIL's default 178 MP limit
        arr = np.array(Image.open(mask_path))
        if arr.ndim == 3:
            # Use alpha only if it actually encodes the mask (>1% transparent)
            if arr.shape[2] == 4 and (arr[:, :, 3] == 0).sum() / arr[:, :, 3].size > 0.01:
                return arr[:, :, 3] > 0
            return arr[:, :, 0] > 0
        return arr > 0
    with rasterio.open(mask_path) as src:
        arr = src.read(4) if src.count == 4 else src.read(1)
    return arr > 0


def find_mask(mask_dir: Path, sheet_id: str) -> Path | None:
    for ext in (".png", ".tif", ".tiff"):
        p = mask_dir / f"{sheet_id}{ext}"
        if p.exists():
            return p
    return None


# Raw map formats, in resolution priority: georeferenced/wrapped forms first,
# plain images last. GDAL reads world-file + .prj sidecars for jpg/png/tif.
RAW_EXTENSIONS = (".tif", ".tiff", ".vrt", ".jpg", ".jpeg", ".png")


def find_raw(raw_root: Path, sheet_id: str) -> Path | None:
    """Return data/raw/<sheet>/<sheet>.<ext> for the first supported extension."""
    for ext in RAW_EXTENSIONS:
        p = raw_root / sheet_id / f"{sheet_id}{ext}"
        if p.exists():
            return p
    return None


def to_pil(data: np.ndarray) -> Image.Image:
    """Convert rasterio (bands, H, W) uint8 array to PIL Image."""
    if data.shape[0] == 1:
        return Image.fromarray(data[0], mode="L")
    return Image.fromarray(np.moveaxis(data, 0, -1))


def patchify(sheet_id: str, require_mask: bool, repo_root: Path):
    cfg = load_config(repo_root)
    pc = cfg["patchify"]
    size, overlap, min_cov, pad = (
        int(pc["patch_size"]),
        int(pc["overlap"]),
        float(pc["min_mask_coverage"]),
        int(pc["pad_value"]),
    )

    raw_root = repo_root / cfg["paths"]["raw"]
    mask_dir = repo_root / cfg["paths"]["masks"] / sheet_id
    out_imgs = repo_root / cfg["paths"]["patches"] / "images" / sheet_id
    out_meta = repo_root / cfg["paths"]["patches"] / "metadata"

    raw_path = find_raw(raw_root, sheet_id)
    if raw_path is None:
        exts = "/".join(e.lstrip(".") for e in RAW_EXTENSIONS)
        sys.exit(f"Raw map not found: {raw_root / sheet_id / sheet_id}.({exts})")

    mask_path = find_mask(mask_dir, sheet_id)
    if require_mask and mask_path is None:
        sys.exit(f"No mask found for '{sheet_id}' in {mask_dir} (.png/.tif/.tiff)")

    use_mask = mask_path is not None

    out_imgs.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)

    with rasterio.open(raw_path) as src:
        if src.dtypes[0] != "uint8":
            sys.exit(f"Expected uint8, got {src.dtypes[0]}. Convert source to uint8 first.")

        img_w, img_h = src.width, src.height
        has_georef = src.crs is not None
        crs = src.crs.to_string() if has_georef else ""
        base_tf = src.transform

        # Scanned maps are RGB or grayscale; drop a 4th (alpha) band from RGBA PNGs
        # so every saved patch is 1- or 3-band.
        n_bands = min(src.count, 3) if src.count == 4 else src.count

        # JPG/PNG have no internal tiling — windowed reads re-decode the whole file
        # each time, so read once into memory and slice.  GeoTIFF/VRT stay windowed.
        in_memory = src.driver in ("JPEG", "PNG")
        full_img = src.read(indexes=list(range(1, n_bands + 1))) if in_memory else None

        print(f"Sheet : {sheet_id}  |  {img_w}x{img_h}px  |  {src.count} band(s)"
              + (f" -> {n_bands}" if n_bands != src.count else "")
              + f"  |  {src.driver}" + ("  (read into memory)" if in_memory else ""))
        print(f"CRS   : {crs or 'none (pixel coordinates)'}")
        if use_mask:
            mask_source = "required (--mask)" if require_mask else "auto-detected"
        else:
            mask_source = "none found"
        print(f"Params: size={size}px  overlap={overlap}px  mask={mask_source}")

        mask_arr = None
        if use_mask:
            mask_arr = load_mask(mask_path)
            print(f"Mask  : {mask_path.name}  |  {100 * mask_arr.sum() / mask_arr.size:.1f}% map area")

        grid = patch_grid(img_w, img_h, size, overlap)
        step = size - overlap
        n_cols = len(range(0, img_w, step))
        print(f"Grid  : {len(grid)} candidate patches")

        records, saved = [], 0

        for idx, (col_off, row_off) in enumerate(tqdm(grid, unit="patch")):
            pw = min(size, img_w - col_off)
            ph = min(size, img_h - row_off)

            if use_mask:
                tile = mask_arr[row_off:row_off + ph, col_off:col_off + pw]
                if tile.sum() / (size * size) < min_cov:
                    continue

            if full_img is not None:
                data = full_img[:, row_off:row_off + ph, col_off:col_off + pw]
            else:
                data = src.read(indexes=list(range(1, n_bands + 1)),
                                window=Window(col_off, row_off, pw, ph))

            if pw < size or ph < size:
                padded = np.full((n_bands, size, size), pad, dtype=data.dtype)
                padded[:, :ph, :pw] = data
                data = padded

            grid_row, grid_col = idx // n_cols, idx % n_cols
            patch_id = f"{sheet_id}_r{grid_row:04d}_c{grid_col:04d}"

            to_pil(data).save(out_imgs / f"{patch_id}.png")

            tf = patch_transform(base_tf, col_off, row_off)
            records.append({
                "patch_id": patch_id, "sheet_id": sheet_id,
                "col_off": col_off, "row_off": row_off,
                "patch_w": pw, "patch_h": ph,
                "grid_col": grid_col, "grid_row": grid_row,
                "has_georef": has_georef, "crs": crs,
                "tf_c": tf.c if has_georef else "", "tf_a": tf.a if has_georef else "",
                "tf_b": tf.b if has_georef else "", "tf_f": tf.f if has_georef else "",
                "tf_d": tf.d if has_georef else "", "tf_e": tf.e if has_georef else "",
            })
            saved += 1

    pd.DataFrame(records).to_csv(out_meta / f"{sheet_id}_patches.csv", index=False)

    print(f"\nSaved {saved} patches  ({len(grid) - saved} skipped)")
    print(f"  -> {out_imgs}/")
    print(f"  -> {out_meta}/{sheet_id}_patches.csv")
    print(
        f"\nNext step: annotate a sample of patches in labelme\n"
        f"  conda activate maptools\n"
        f"  python steps/02_annotate/annotate.py --sheet {sheet_id}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patchify a map sheet into 512px PNG tiles.")
    parser.add_argument("--sheet", required=True, help="Sheet ID (subfolder name under data/raw/)")
    parser.add_argument(
        "--mask", action="store_true",
        help="Require a map-area mask and exit if none is found. "
             "Without this flag a mask is still used automatically if one exists.",
    )
    args = parser.parse_args()

    patchify(args.sheet, args.mask, Path(__file__).resolve().parents[2])
