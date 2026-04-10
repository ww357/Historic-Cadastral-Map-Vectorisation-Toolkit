"""
Slice a georeferenced map sheet into 1024px PNG patches.

With --mask, only patches that overlap the map-area mask by at least
min_mask_coverage (config.yaml) are saved. All spatial metadata needed
for stitching predictions back into a full-document raster is written
to a CSV alongside the patches.

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


def to_pil(data: np.ndarray) -> Image.Image:
    """Convert rasterio (bands, H, W) uint8 array to PIL Image."""
    if data.shape[0] == 1:
        return Image.fromarray(data[0], mode="L")
    return Image.fromarray(np.moveaxis(data, 0, -1))


def patchify(sheet_id: str, use_mask: bool, repo_root: Path):
    cfg = load_config(repo_root)
    pc = cfg["patchify"]
    size, overlap, min_cov, pad = (
        int(pc["patch_size"]),
        int(pc["overlap"]),
        float(pc["min_mask_coverage"]),
        int(pc["pad_value"]),
    )

    raw_path = repo_root / cfg["paths"]["raw"] / sheet_id / f"{sheet_id}.tif"
    mask_dir = repo_root / cfg["paths"]["masks"] / sheet_id
    out_imgs = repo_root / cfg["paths"]["patches"] / "images" / sheet_id
    out_meta = repo_root / cfg["paths"]["patches"] / "metadata"

    if not raw_path.exists():
        sys.exit(f"Raw map not found: {raw_path}")

    mask_path = find_mask(mask_dir, sheet_id)
    if use_mask and mask_path is None:
        sys.exit(f"No mask found for '{sheet_id}' in {mask_dir} (.png/.tif/.tiff)")

    out_imgs.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)

    with rasterio.open(raw_path) as src:
        if src.dtypes[0] != "uint8":
            sys.exit(f"Expected uint8, got {src.dtypes[0]}. Convert source to uint8 first.")

        img_w, img_h = src.width, src.height
        has_georef = src.crs is not None
        crs = src.crs.to_string() if has_georef else ""
        base_tf = src.transform

        print(f"Sheet : {sheet_id}  |  {img_w}x{img_h}px  |  {src.count} band(s)")
        print(f"CRS   : {crs or 'none'}")
        print(f"Params: size={size}px  overlap={overlap}px  mask={'yes' if use_mask else 'no'}")

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

            data = src.read(window=Window(col_off, row_off, pw, ph))

            if pw < size or ph < size:
                padded = np.full((src.count, size, size), pad, dtype=data.dtype)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patchify a map sheet into 1024px PNG tiles.")
    parser.add_argument("--sheet", required=True, help="Sheet ID (subfolder name under data/raw/)")
    parser.add_argument("--mask", action="store_true", help="Skip patches outside the map-area mask")
    args = parser.parse_args()

    patchify(args.sheet, args.mask, Path(__file__).resolve().parents[2])
