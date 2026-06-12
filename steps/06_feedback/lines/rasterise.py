"""
Rasterise corrected boundary vectors from GeoPackage back to 256×256 training tiles,
then register them in the dataset manifest for feedback fine-tuning.

Two sources of new tiles are added to the training dataset in one pass:

  1. ANNOTATION tiles (ground_truth): hand-drawn masks from data/annotations/boundary/
     for the current sheet that are not yet in the manifest.  These are full-weight
     training examples — equivalent in quality to the original training data.

  2. FEEDBACK tiles (pseudo_label): rasterised from the mended 'boundaries' layer in
     the GeoPackage.  Only patches where the U-Net prediction was used are eligible
     (hand-annotated patches are excluded — rasterising them via the vector round-trip
     would degrade clean ground truth through simplification artefacts).

For each eligible feedback patch:
  1. Clips the mended 'boundaries' GeoDataFrame to the patch extent
  2. Buffers LineStrings by line_width/2 CRS units to reconstruct line width
  3. Rasterises to a 512×512 binary mask
  4. Splits to four 256×256 tiles matching the training dataset layout
  5. Saves image and mask tiles (suffix _fb) to the training dataset directories
  6. Appends manifest entries with source=feedback, tier=pseudo_label

Reads  : data/outputs/<SHEET_ID>.gpkg            — mended boundaries layer
         data/patches/metadata/<SHEET_ID>_patches.csv
         data/predictions/boundaries/<SHEET_ID>/  — eligible patch check
         data/patches/images/<SHEET_ID>/           — source image patches
         data/annotations/boundary/<SHEET_ID>/masks/  — annotation masks (if any)
         data/raw/<SHEET_ID>/<SHEET_ID>.tif        — georef transform

Writes : data/training/boundary_dataset/train/             — new 256px image tiles
         data/training/boundary_dataset/annotation/train/  — new 256px mask tiles
         data/training/boundary_dataset/manifest.csv       — updated manifest
         data/feedback/boundary/<SHEET_ID>/eligible.csv    — per-patch eligibility log

Usage:
    conda activate maptools
    python steps/06_feedback/lines/rasterise.py --sheet SHEET_ID

Then run train.py in the tf-gpu environment:
    conda activate tf-gpu
    python steps/06_feedback/lines/train.py --sheet SHEET_ID --name feedback_v1
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import Affine
import yaml
from PIL import Image
from shapely.geometry import box
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]

MANIFEST_COLS = ["split", "image_path", "mask_path", "sheet", "source", "tier"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    p = ROOT / "config.yaml"
    if not p.exists():
        sys.exit(f"config.yaml not found at {p}")
    return yaml.safe_load(p.read_text())


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _extract_sheet_from_stem(stem: str) -> str:
    """Extract sheet name from a tile filename stem.

    Handles:
      {Sheet}_{col}_{row}.png       — original training tiles
      {Sheet}_{col}_{row}_fb.png    — feedback tiles
      {Sheet}_{col}_{row}_ann.png   — annotation tiles
    """
    parts = stem.split("_")
    if parts and parts[-1] in ("fb", "ann"):
        parts = parts[:-1]
    # Walk backwards to find the last two consecutive integer segments
    for i in range(len(parts) - 1, 0, -1):
        if parts[i].isdigit() and parts[i - 1].isdigit():
            return "_".join(parts[: i - 1])
    return stem


def _classify_stem(stem: str) -> tuple[str, str]:
    """Return (source, tier) based on filename suffix."""
    if stem.endswith("_fb"):
        return "feedback", "pseudo_label"
    if stem.endswith("_ann"):
        return "annotation", "ground_truth"
    return "original", "ground_truth"


def bootstrap_manifest(dataset_dir: Path) -> list[dict]:
    """Build a manifest from all existing tiles in the training dataset directory.
    Called once if manifest.csv does not yet exist.
    """
    rows = []
    for split in ("train", "test"):
        img_dir = dataset_dir / split
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.png")):
            stem = img_path.stem
            sheet = _extract_sheet_from_stem(stem)
            source, tier = _classify_stem(stem)
            rows.append({
                "split":      split,
                "image_path": f"{split}/{img_path.name}",
                "mask_path":  f"annotation/{split}/{img_path.name}",
                "sheet":      sheet,
                "source":     source,
                "tier":       tier,
            })
    return rows


def _write_manifest(manifest_path: Path, rows: list[dict]) -> None:
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        writer.writeheader()
        writer.writerows(rows)


def load_or_create_manifest(dataset_dir: Path) -> tuple[list[dict], set[str]]:
    """Load manifest.csv, or bootstrap from existing files if absent.
    Returns (rows, existing_image_path_set).
    """
    manifest_path = dataset_dir / "manifest.csv"
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        rows = df.to_dict("records")
        print(f"Manifest loaded: {len(rows)} existing entries")
    else:
        print("manifest.csv not found — bootstrapping from existing training data...")
        rows = bootstrap_manifest(dataset_dir)
        _write_manifest(manifest_path, rows)
        print(f"  Created manifest.csv: {len(rows)} existing tiles catalogued")
    return rows, {r["image_path"] for r in rows}


# ---------------------------------------------------------------------------
# Rasterisation helpers
# ---------------------------------------------------------------------------

def _local_transform(
    col_off: int, row_off: int,
    img_transform: Affine | None,
    has_georef: bool,
) -> Affine:
    """Affine transform with origin at the patch top-left corner."""
    if has_georef and img_transform is not None:
        px, py = img_transform * (col_off, row_off)
        return Affine(img_transform.a, img_transform.b, px,
                      img_transform.d, img_transform.e, py)
    # Non-georef: pixel-space identity with patch offset
    return Affine(1.0, 0.0, float(col_off), 0.0, 1.0, float(row_off))


def _patch_bbox(local_tf: Affine, patch_w: int, patch_h: int):
    """Shapely box covering the patch extent in map CRS (or pixel) coordinates."""
    x0, y0 = local_tf * (0,       0)
    x1, y1 = local_tf * (patch_w, patch_h)
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _rasterize_to_mask(
    gdf: gpd.GeoDataFrame,
    bbox,
    local_tf: Affine,
    patch_size: int,
    line_width_px: int,
) -> np.ndarray:
    """
    Clip boundary LineStrings to bbox, buffer by line_width_px/2 CRS units,
    and rasterise onto a (patch_size × patch_size) binary uint8 canvas.
    """
    candidates = gdf[gdf.geometry.intersects(bbox)]
    if candidates.empty:
        return np.zeros((patch_size, patch_size), dtype=np.uint8)

    clipped_geoms = [
        geom.intersection(bbox)
        for geom in candidates.geometry
        if geom is not None and not geom.is_empty
    ]
    clipped_geoms = [g for g in clipped_geoms if g is not None and not g.is_empty]
    if not clipped_geoms:
        return np.zeros((patch_size, patch_size), dtype=np.uint8)

    # Buffer in CRS units: pixel_size = |tf.a|, buffer = (line_width/2) × pixel_size
    buffer_dist = (line_width_px / 2.0) * abs(local_tf.a)
    burn_shapes = [
        (geom.buffer(buffer_dist), 1)
        for geom in clipped_geoms
        if not geom.buffer(buffer_dist).is_empty
    ]
    if not burn_shapes:
        return np.zeros((patch_size, patch_size), dtype=np.uint8)

    return rio_rasterize(
        burn_shapes,
        out_shape=(patch_size, patch_size),
        transform=local_tf,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )


# ---------------------------------------------------------------------------
# Tile saving utility
# ---------------------------------------------------------------------------

def _save_tiles(
    src_img_512: np.ndarray,
    mask_512: np.ndarray,
    col_off: int,
    row_off: int,
    tile_size: int,
    patch_w: int,
    patch_h: int,
    sheet_id: str,
    suffix: str,
    dataset_dir: Path,
    existing_paths: set[str],
    new_rows: list[dict],
    rng: np.random.Generator,
    test_split: float,
) -> int:
    """Split 512px arrays into tile_size tiles, save, and append manifest entries.
    Returns number of new tiles added.
    """
    n_added = 0
    n_tiles_x = patch_w // tile_size
    n_tiles_y = patch_h // tile_size

    for tr in range(n_tiles_y):
        for tc in range(n_tiles_x):
            rs, re = tr * tile_size, (tr + 1) * tile_size
            cs, ce = tc * tile_size, (tc + 1) * tile_size

            tile_img  = src_img_512[rs:re, cs:ce]
            tile_mask = mask_512   [rs:re, cs:ce]

            global_col = (col_off + cs) // tile_size
            global_row = (row_off + rs) // tile_size
            tile_fname = f"{sheet_id}_{global_col}_{global_row}_{suffix}.png"

            split   = "test" if rng.random() < test_split else "train"
            img_rel = f"{split}/{tile_fname}"
            msk_rel = f"annotation/{split}/{tile_fname}"

            if img_rel in existing_paths:
                continue

            Image.fromarray(tile_img,  mode="L").save(dataset_dir / split          / tile_fname)
            Image.fromarray(tile_mask, mode="L").save(dataset_dir / "annotation" / split / tile_fname)

            tier   = "ground_truth" if suffix == "ann" else "pseudo_label"
            source = "annotation"   if suffix == "ann" else "feedback"
            new_rows.append({
                "split":      split,
                "image_path": img_rel,
                "mask_path":  msk_rel,
                "sheet":      sheet_id,
                "source":     source,
                "tier":       tier,
            })
            existing_paths.add(img_rel)
            n_added += 1

    return n_added


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rasterise mended boundary vectors to 256px training tiles."
    )
    parser.add_argument("--sheet",      required=True,
                        help="Sheet ID (must match folder names in data/)")
    parser.add_argument("--test-split", type=float, default=0.15,
                        help="Fraction of new tiles assigned to test split (default: 0.15)")
    args = parser.parse_args()

    sheet_id   = args.sheet
    test_split = args.test_split

    cfg     = load_config()
    paths   = cfg["paths"]
    ann_cfg = cfg["annotation"]

    line_width = int(ann_cfg.get("line_width", 3))
    patch_size = int(cfg["patchify"]["patch_size"])   # 512
    tile_size  = int(cfg["unet"]["inference_size"])   # 256

    boundary_label = ann_cfg.get("boundary_label", "boundary")

    dataset_dir  = ROOT / "data" / "training" / "boundary_dataset"
    gpkg_path    = ROOT / paths["outputs"]     / f"{sheet_id}.gpkg"
    meta_path    = ROOT / paths["patches"]     / "metadata" / f"{sheet_id}_patches.csv"
    pred_dir     = ROOT / paths["predictions"] / "boundaries" / sheet_id
    img_dir      = ROOT / paths["patches"]     / "images"    / sheet_id
    ann_mask_dir = ROOT / paths["annotations"] / boundary_label / sheet_id / "masks"
    raw_path     = ROOT / paths["raw"]         / sheet_id / f"{sheet_id}.tif"
    feedback_dir = ROOT / "data" / "feedback"  / "boundary" / sheet_id

    feedback_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["train", "test", "annotation/train", "annotation/test"]:
        (dataset_dir / subdir).mkdir(parents=True, exist_ok=True)

    # ---- Validate -----------------------------------------------------------
    for p, label in [(gpkg_path, "GeoPackage"), (meta_path, "Patch metadata")]:
        if not p.exists():
            sys.exit(f"{label} not found: {p}")

    # ---- Manifest -----------------------------------------------------------
    manifest_rows, existing_paths = load_or_create_manifest(dataset_dir)

    # ---- Boundaries GeoPackage ---------------------------------------------
    try:
        boundaries_gdf = gpd.read_file(gpkg_path, layer="boundaries")
    except Exception as e:
        sys.exit(f"Could not read 'boundaries' layer from {gpkg_path}:\n  {e}")

    if boundaries_gdf.empty:
        print("Warning: 'boundaries' layer is empty. Feedback masks will all be blank.")

    # ---- Georef from raw TIF ------------------------------------------------
    img_transform: Affine | None = None
    has_georef = False
    if raw_path.exists():
        with rasterio.open(raw_path) as src:
            has_georef    = src.crs is not None
            img_transform = src.transform if has_georef else None
    else:
        print(f"Warning: raw TIF not found at {raw_path} — assuming non-georeferenced.")

    # ---- Patch metadata -----------------------------------------------------
    meta = pd.read_csv(meta_path)

    # ---- Eligible patches: predicted (not annotated) ------------------------
    eligible_fb  = []   # for feedback rasterisation
    eligible_ann = []   # for annotation tile export
    for _, row in meta.iterrows():
        ann_exists  = (ann_mask_dir / f"{row.patch_id}.png").exists()
        pred_exists = (pred_dir     / f"{row.patch_id}.png").exists()
        img_exists  = (img_dir      / f"{row.patch_id}.png").exists()
        if not img_exists:
            continue
        if ann_exists:
            eligible_ann.append(row)
        elif pred_exists:
            eligible_fb.append(row)

    print(f"\n── Rasterise feedback ─────────────────────────────────────")
    print(f"Sheet      : {sheet_id}  |  Georef: {has_georef}")
    print(f"Feedback   : {len(eligible_fb)} eligible patches "
          f"(predicted, not hand-annotated)")
    print(f"Annotation : {len(eligible_ann)} patches with hand-drawn masks")
    print(f"Boundaries : {len(boundaries_gdf)} features in GeoPackage")
    print(f"Line width : {line_width}px → buffer {line_width / 2:.1f} "
          f"{'CRS units' if has_georef else 'pixels'}")

    rng = np.random.default_rng(42)
    new_rows: list[dict]         = []
    eligible_records: list[dict] = []
    n_fb_added = n_ann_added = n_existed = 0

    # =========================================================================
    # A) Annotation tiles — split existing 512px hand-drawn masks to 256px tiles
    # =========================================================================
    if eligible_ann:
        print(f"\nAdding annotation tiles...")
        for row in tqdm(eligible_ann, unit="patch", desc="Annotations"):
            col_off, row_off = int(row.col_off), int(row.col_off)
            col_off, row_off = int(row.col_off), int(row.row_off)
            pw, ph = int(row.patch_w), int(row.patch_h)

            src_img  = np.array(Image.open(img_dir / f"{row.patch_id}.png").convert("L"),
                                dtype=np.uint8)
            ann_mask = np.array(Image.open(ann_mask_dir / f"{row.patch_id}.png").convert("L"),
                                dtype=np.uint8)
            # Binarise annotation mask
            ann_mask = (ann_mask > 127).astype(np.uint8) * 255

            added = _save_tiles(
                src_img, ann_mask, col_off, row_off, tile_size, pw, ph,
                sheet_id, "ann", dataset_dir, existing_paths, new_rows, rng, test_split,
            )
            n_ann_added += added
            n_existed   += (pw // tile_size) * (ph // tile_size) - added

    # =========================================================================
    # B) Feedback tiles — rasterise mended vectors onto each eligible patch
    # =========================================================================
    print(f"\nRasterising feedback patches...")
    for row in tqdm(eligible_fb, unit="patch", desc="Feedback"):
        col_off, row_off = int(row.col_off), int(row.row_off)
        pw, ph = int(row.patch_w), int(row.patch_h)

        local_tf = _local_transform(col_off, row_off, img_transform, has_georef)
        bbox     = _patch_bbox(local_tf, pw, ph)

        mask_512 = _rasterize_to_mask(
            boundaries_gdf, bbox, local_tf, patch_size, line_width,
        )
        n_foreground = int((mask_512 > 0).sum())

        src_img = np.array(
            Image.open(img_dir / f"{row.patch_id}.png").convert("L"), dtype=np.uint8
        )

        added = _save_tiles(
            src_img, mask_512, col_off, row_off, tile_size, pw, ph,
            sheet_id, "fb", dataset_dir, existing_paths, new_rows, rng, test_split,
        )
        n_fb_added += added
        n_existed  += (pw // tile_size) * (ph // tile_size) - added

        eligible_records.append({
            "patch_id":        row.patch_id,
            "n_foreground_px": n_foreground,
            "n_tiles_added":   added,
        })

    # ---- Write manifest -----------------------------------------------------
    all_rows = manifest_rows + new_rows
    _write_manifest(dataset_dir / "manifest.csv", all_rows)

    # ---- Write eligible log -------------------------------------------------
    if eligible_records:
        with open(feedback_dir / "eligible.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["patch_id", "n_foreground_px", "n_tiles_added"]
            )
            writer.writeheader()
            writer.writerows(eligible_records)

    # ---- Summary ------------------------------------------------------------
    print(f"\n── Results ────────────────────────────────────────────────")
    print(f"Annotation tiles added : {n_ann_added}")
    print(f"Feedback tiles added   : {n_fb_added}")
    print(f"Already in dataset     : {n_existed}  (skipped)")
    print(f"Manifest total         : {len(all_rows)} entries")
    print(f"\nManifest  → {(dataset_dir / 'manifest.csv').relative_to(ROOT)}")
    if eligible_records:
        print(f"Eligible  → {(feedback_dir / 'eligible.csv').relative_to(ROOT)}")
    print(f"\nNext step:")
    print(f"  conda activate tf-gpu")
    print(f"  python steps/06_feedback/lines/train.py "
          f"--sheet {sheet_id} --name feedback_v1")


if __name__ == "__main__":
    main()
