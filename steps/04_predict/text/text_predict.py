"""
Run MapTextPipeline text spotting on all patches for a given sheet.

Uses MapReader's MapTextRunner to:
  1. Run detection + recognition on every 512px patch
  2. Deduplicate overlapping boxes at patch level (IoA filter)
  3. Scale predictions up to parent-image pixel coordinates
  4. Deduplicate again across patch boundaries
  5. Georeference to map CRS (if sheet has a georef transform)
  6. Save to data/predictions/text/<SHEET_ID>/text_preds.geojson
  7. Add a "text" layer to data/outputs/<SHEET_ID>.gpkg

Reads  : data/patches/images/<SHEET_ID>/*.png
         data/patches/metadata/<SHEET_ID>_patches.csv
         data/raw/<SHEET_ID>/<SHEET_ID>.tif  (for full dimensions + georef)

Writes : data/predictions/text/<SHEET_ID>/text_preds.geojson
         data/outputs/<SHEET_ID>.gpkg        (layer "text")

Usage:
    python steps/04_predict/text/text_predict.py --sheet SHEET_ID [--weights path/to/weights.pth] [--device cpu]
    python steps/04_predict/polygons/predict.py --sheet SHEET_ID --feature text [water building ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    p = ROOT / "config.yaml"
    if not p.exists():
        sys.exit(f"config.yaml not found at {p}")
    return yaml.safe_load(p.read_text())


# ---------------------------------------------------------------------------
# Weight resolution
# ---------------------------------------------------------------------------

def resolve_weights(weights_arg: str | None, repo_root: Path, paths_cfg: dict) -> Path:
    """
    Search order:
      1. --weights CLI argument
      2. models/base/MapTextPipeline/*.pth  (most recently modified)
      3. models/finetuned/*text*.pth        (most recently modified)
    """
    if weights_arg:
        p = Path(weights_arg)
        if not p.is_absolute():
            p = repo_root / p
        if p.exists():
            return p
        sys.exit(f"Weights not found: {p}")

    base_dir = repo_root / paths_cfg["models_base"] / "MapTextPipeline"
    candidates = sorted(base_dir.glob("*.pth"), key=lambda x: x.stat().st_mtime)
    if candidates:
        return candidates[-1]

    finetuned_dir = repo_root / paths_cfg["models_finetuned"]
    candidates = sorted(finetuned_dir.glob("*text*.pth"), key=lambda x: x.stat().st_mtime)
    if candidates:
        return candidates[-1]

    sys.exit(
        f"No text model weights found.\n"
        f"  Searched: {base_dir}/*.pth\n"
        f"  Searched: {finetuned_dir}/*text*.pth\n"
        "  Place rumsey-finetune.pth in models/base/MapTextPipeline/ and try again.\n"
        "  Or pass --weights <path> to specify a file explicitly."
    )


# ---------------------------------------------------------------------------
# DataFrame construction  (our CSV format → MapReader format)
# ---------------------------------------------------------------------------

def build_patch_df(meta: pd.DataFrame, patches_dir: Path) -> tuple[pd.DataFrame, int]:
    """
    Convert our patches metadata CSV into the patch_df format MapTextRunner expects.

    MapReader patch_df requires:
      index       : image filename  (e.g. "Timberscombe_r0003_c0006.png")
      image_path  : absolute path to patch PNG
      parent_id   : sheet ID — links to parent_df index
      pixel_bounds: (col_off, row_off, col_off+patch_w, row_off+patch_h)
                    i.e. (min_x, min_y, max_x, max_y) in parent pixel space
    """
    records = []
    for _, row in meta.iterrows():
        img_path = patches_dir / f"{row.patch_id}.png"
        if not img_path.exists():
            continue
        col_off  = int(row.col_off)
        row_off  = int(row.row_off)
        patch_w  = int(row.patch_w)
        patch_h  = int(row.patch_h)
        records.append({
            "image_path":   str(img_path),
            "parent_id":    str(row.sheet_id),
            "pixel_bounds": (col_off, row_off, col_off + patch_w, row_off + patch_h),
        })

    if not records:
        return pd.DataFrame(), 0

    df = pd.DataFrame(records)
    df.index = [Path(r["image_path"]).name for r in records]
    return df, len(records)


def build_parent_df(
    sheet_id: str,
    raw_path: Path,
    meta: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    """
    Build the parent_df MapTextRunner expects for georeferencing.

    MapReader parent_df requires:
      index       : sheet ID (must match patch_df["parent_id"])
      image_path  : path to full-sheet image (used for display only)
      coordinates : (left, bottom, right, top) in CRS units
      dlon        : x increment per pixel  = tf_a from Affine
      dlat        : y increment per pixel  = abs(tf_e) from Affine
      crs         : CRS string (e.g. "EPSG:27700")

    MapReader's convert_to_coords formula:
        geo_x = pixel_x * dlon + coordinates[0]   (coordinates[0] = left)
        geo_y = coordinates[3] - pixel_y * dlat    (coordinates[3] = top)
    which matches a north-up Affine: geo_x = tf_c + px*tf_a,  geo_y = tf_f + py*tf_e
    """
    # has_georef is written as a Python bool then round-tripped through CSV,
    # so compare as string to avoid bool("False") == True.
    row0 = meta.iloc[0]
    has_georef = str(row0.has_georef).strip().lower() == "true"

    if raw_path.exists():
        with rasterio.open(raw_path) as src:
            img_w, img_h = src.width, src.height
            # Read georef directly from the raw TIF — never from meta.iloc[0].
            # When --mask is used, the first CSV row is the first patch that
            # passed the mask filter, which may be far from pixel (0, 0).
            # Using its tf_c/tf_f as the image origin introduces a systematic
            # coordinate offset equal to that patch's pixel position × pixel size.
            if src.crs is not None:
                has_georef = True
                base_tf = src.transform
                tf_a    = base_tf.a        # x pixel size  (+ve)
                tf_e    = base_tf.e        # y pixel size  (-ve for north-up)
                tf_c    = base_tf.c        # easting  of true image top-left
                tf_f    = base_tf.f        # northing of true image top-left
                crs_str = src.crs.to_string()
            else:
                has_georef = False
                crs_str    = None
    else:
        # Raw TIF unavailable — reconstruct origin from any patch's stored
        # transform by back-calculating to pixel (0, 0).
        img_w = int((meta["col_off"] + meta["patch_w"]).max())
        img_h = int((meta["row_off"] + meta["patch_h"]).max())
        r     = meta.sort_values(["col_off", "row_off"]).iloc[0]
        tf_a  = float(r.tf_a)
        tf_e  = float(r.tf_e)
        tf_c  = float(r.tf_c) - int(r.col_off) * tf_a   # back to image origin
        tf_f  = float(r.tf_f) - int(r.row_off) * tf_e
        crs_str = str(r.crs) if has_georef else None

    if has_georef:
        left   = tf_c
        top    = tf_f
        right  = tf_c + img_w * tf_a
        bottom = tf_f + img_h * tf_e   # tf_e negative → bottom < top
        dlon   = tf_a
        dlat   = abs(tf_e)
    else:
        left, bottom = 0.0, float(img_h)
        right, top   = float(img_w), 0.0
        dlon = dlat  = 1.0
        crs_str      = None

    parent_df = pd.DataFrame(
        [{
            "image_path":   str(raw_path) if raw_path.exists() else "",
            "coordinates":  (left, bottom, right, top),
            "dlon":         dlon,
            "dlat":         dlat,
            "crs":          crs_str,
        }],
        index=[sheet_id],
    )
    return parent_df, has_georef


# ---------------------------------------------------------------------------
# Tile assembly
# ---------------------------------------------------------------------------

def _group_into_tiles(meta: pd.DataFrame, tile_n: int) -> list[pd.DataFrame]:
    """Group existing patches into tile_n × tile_n grids by grid position."""
    m = meta.copy()
    m["_tile_row"] = m["grid_row"] // tile_n
    m["_tile_col"] = m["grid_col"] // tile_n
    return [grp for _, grp in m.groupby(["_tile_row", "_tile_col"], sort=True)]


def _assemble_tile(
    group: pd.DataFrame,
    patches_dir: Path,
    pad_value: int = 255,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """
    Stitch a group of patches into one RGB tile image.

    Returns:
        tile image, (min_col, min_row, max_col, max_row) in parent pixels.
    Missing patches are filled with pad_value (white).
    """
    min_col = int(group["col_off"].min())
    min_row = int(group["row_off"].min())
    max_col = int((group["col_off"] + group["patch_w"]).max())
    max_row = int((group["row_off"] + group["patch_h"]).max())
    tile_w  = max_col - min_col
    tile_h  = max_row - min_row

    tile = Image.new("RGB", (tile_w, tile_h), color=(pad_value, pad_value, pad_value))
    for _, row in group.iterrows():
        p = patches_dir / f"{row.patch_id}.png"
        if not p.exists():
            continue
        patch = Image.open(p).convert("RGB")
        tile.paste(patch, (int(row.col_off) - min_col, int(row.row_off) - min_row))
    return tile, (min_col, min_row, max_col, max_row)


def build_tile_df(
    meta: pd.DataFrame,
    patches_dir: Path,
    tiles_dir: Path,
    tile_n: int,
    pad_value: int = 255,
) -> tuple[pd.DataFrame, int, int]:
    """
    Assemble tile_n × tile_n patch grids into tile PNG files and return a
    patch_df that points to tiles instead of individual patches.

    MapReader uses pixel_bounds to map tile predictions back to parent pixel
    coordinates. Because each tile pixel corresponds exactly to one parent
    pixel (no resampling), the mapping is a simple offset — transparent to
    MapTextRunner regardless of how many patches are in each tile.

    Returns:
        (tile_df, n_tiles, n_patches_covered)
    """
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # Only group patches that actually exist on disk
    existing = meta[
        meta["patch_id"].apply(lambda pid: (patches_dir / f"{pid}.png").exists())
    ].copy()
    if existing.empty:
        return pd.DataFrame(), 0, 0

    groups  = _group_into_tiles(existing, tile_n)
    records = []
    for i, group in enumerate(groups):
        tile_img, bounds = _assemble_tile(group, patches_dir, pad_value)
        tile_path = tiles_dir / f"tile_{i:04d}.png"
        tile_img.save(tile_path)
        records.append({
            "image_path":   str(tile_path),
            "parent_id":    str(group["sheet_id"].iloc[0]),
            "pixel_bounds": bounds,
        })

    df = pd.DataFrame(records)
    df.index = [Path(r["image_path"]).name for r in records]
    return df, len(records), len(existing)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def predict(sheet_id: str, repo_root: Path,
            weights_arg: str | None = None,
            device_arg:  str | None = None):

    cfg       = load_config()
    paths_cfg = cfg["paths"]
    text_cfg  = cfg.get("text", {})

    # Paths
    patches_dir  = repo_root / paths_cfg["patches"] / "images"  / sheet_id
    meta_path    = repo_root / paths_cfg["patches"] / "metadata" / f"{sheet_id}_patches.csv"
    raw_path     = repo_root / paths_cfg["raw"]     / sheet_id  / f"{sheet_id}.tif"
    pred_dir     = repo_root / paths_cfg["predictions"] / "text" / sheet_id
    outputs_dir  = repo_root / paths_cfg["outputs"]
    gpkg_path    = outputs_dir / f"{sheet_id}.gpkg"

    if not patches_dir.exists():
        sys.exit(f"Patches not found: {patches_dir}  — run 01_patchify first.")
    if not meta_path.exists():
        sys.exit(f"Metadata CSV not found: {meta_path}  — run 01_patchify first.")

    pred_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    geojson_path     = pred_dir / "text_preds.geojson"
    checkpoint_path  = pred_dir / "checkpoints" / "patch_predictions.csv"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Fast-exit: GeoJSON already saved — just need text_to_vector.py
    # ------------------------------------------------------------------
    if geojson_path.exists():
        print(f"GeoJSON already exists — inference and georeferencing are complete.")
        print(f"  {geojson_path.relative_to(repo_root)}")
        print(f"\nTo add the text layer to the GeoPackage:")
        print(f"  conda activate maptools")
        print(f"  python steps/05_vectorise/text/text_to_vector.py --sheet {sheet_id}")
        print(f"\nTo force a full re-run, delete these files first:")
        print(f"  rm {geojson_path}")
        print(f"  rm {checkpoint_path}")
        return

    # Settings — CLI args override config.yaml
    weights_path  = resolve_weights(weights_arg, repo_root, paths_cfg)
    min_ioa       = float(text_cfg.get("min_ioa", 0.7))

    # Resolve device — never pass "default" to MapTextRunner as it may
    # silently fall back to CPU. Resolve to "cuda" or "cpu" explicitly.
    _device_cfg = device_arg or text_cfg.get("device", "default")
    if _device_cfg == "default":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = _device_cfg
    cfg_file_path = repo_root / text_cfg.get(
        "cfg_file",
        "models/MapTextPipeline/configs/ViTAEv2_S/rumsey/final_rumsey.yaml",
    )

    print(f"Sheet    : {sheet_id}")
    print(f"Weights  : {weights_path.relative_to(repo_root)}")
    print(f"Config   : {cfg_file_path.relative_to(repo_root)}")
    print(f"Device   : {device}{' ⚠ no CUDA found — running on CPU' if device == 'cpu' else ''}")
    print(f"Min IoA  : {min_ioa}")
    print()

    # ------------------------------------------------------------------
    # Build DataFrames
    # ------------------------------------------------------------------
    meta = pd.read_csv(meta_path)
    patch_df, n_found = build_patch_df(meta, patches_dir)

    if n_found == 0:
        sys.exit(
            f"No patch images found under {patches_dir}\n"
            "Check that 01_patchify has been run for this sheet."
        )

    parent_df, has_georef = build_parent_df(sheet_id, raw_path, meta)

    print(f"Patches  : {n_found} images found  ({len(meta) - n_found} in CSV but missing on disk)")
    if not has_georef:
        print("Warning  : No georeferencing found — output will use pixel coordinates.")
    print()

    # ------------------------------------------------------------------
    # Import MapReader here so a missing install gives a clear message
    # ------------------------------------------------------------------
    try:
        from mapreader import MapTextRunner
    except ImportError:
        sys.exit(
            "mapreader package not found.\n"
            "Activate the correct environment:  conda activate New-MapReader\n"
            "Then install:  pip install mapreader"
        )

    # ------------------------------------------------------------------
    # Tile assembly (optional, controlled by text.tile_n in config.yaml)
    # ------------------------------------------------------------------
    # MIN_SIZE_TEST: 1024 means a 512px patch is upscaled 2× before every
    # forward pass. Assembling tile_n × tile_n patches into a ~1024px tile
    # first means one forward pass covers tile_n² patches, giving ~3-4×
    # throughput on GPU with no accuracy loss.
    tile_n    = int(text_cfg.get("tile_n", 1))
    pad_value = int(cfg.get("patchify", {}).get("pad_value", 255))
    tiles_dir = pred_dir / "tiles"

    if tile_n > 1:
        print(f"Assembling {tile_n}×{tile_n} patch tiles for inference...")
        inference_df, n_tiles, n_covered = build_tile_df(
            meta, patches_dir, tiles_dir, tile_n, pad_value
        )
        if n_tiles == 0:
            sys.exit("No patch images found to assemble into tiles.")
        print(f"  {n_tiles} tiles  ({n_covered} patches, "
              f"avg {n_covered / n_tiles:.1f} patches/tile)")
    else:
        inference_df = patch_df

    # ------------------------------------------------------------------
    # Instantiate runner
    # ------------------------------------------------------------------
    runner = MapTextRunner(
        inference_df,
        parent_df,
        cfg_file     = str(cfg_file_path),
        weights_file = str(weights_path),
        device       = device,
    )

    # ------------------------------------------------------------------
    # Run inference OR resume from checkpoint
    # Checkpoint is saved immediately after run_all() so a crash in any
    # post-processing step (georeferencing, GeoJSON write) never loses
    # hours of GPU inference — just re-run the same command to resume.
    # ------------------------------------------------------------------
    if checkpoint_path.exists():
        print(f"Checkpoint found — skipping inference, loading saved predictions...")
        runner.load_patch_predictions(str(checkpoint_path))
        # load_patch_predictions() auto-calls convert_to_parent_pixel_bounds();
        # call explicitly with our dedup settings to ensure correct parameters.
        parent_preds_df = runner.convert_to_parent_pixel_bounds(
            return_dataframe=True,
            deduplicate=True,
            min_ioa=min_ioa,
        )
        n_loaded = len(runner.patch_predictions) if runner.patch_predictions else 0
        print(f"  Loaded: {n_loaded} patch-level instances")
        print(f"  Sheet-level instances (after dedup): {len(parent_preds_df)}")
    else:
        print("Running text spotting on all patches...")
        runner.run_all(min_ioa=min_ioa)
        preds_dict    = getattr(runner, "patch_predictions", None) or {}
        n_patch_preds = sum(len(v) for v in preds_dict.values()) if preds_dict else 0
        print(f"  Patch-level instances: {n_patch_preds}")

        # Save checkpoint immediately — before any post-processing that could fail
        runner.to_csv(str(checkpoint_path))
        print(f"  Checkpoint saved → {checkpoint_path.relative_to(repo_root)}")

        print("Scaling up to full-sheet coordinates and deduplicating...")
        parent_preds_df = runner.convert_to_parent_pixel_bounds(
            return_dataframe=True,
            deduplicate=True,
            min_ioa=min_ioa,
        )
        print(f"  Sheet-level instances (after dedup): {len(parent_preds_df)}")

    # ------------------------------------------------------------------
    # Georeference (if CRS available)
    # ------------------------------------------------------------------
    if has_georef:
        print("Georeferencing predictions...")
        # convert_to_coords() must be called before to_geojson() to populate
        # MapReader's internal georeferenced state (raises ValueError without it).
        # We read the GeoJSON back rather than using the return value, which
        # gives clean column dtypes and avoids fiona's NULL pointer error.
        runner.convert_to_coords()
        runner.to_geojson(str(geojson_path))
        print(f"  GeoJSON → {geojson_path.relative_to(repo_root)}")

        geo_gdf = gpd.read_file(geojson_path)
        print(f"  Georeferenced instances: {len(geo_gdf)}")
    else:
        # No CRS — save parent pixel bounds as GeoJSON directly
        geo_gdf = parent_preds_df.copy()
        if not isinstance(geo_gdf, gpd.GeoDataFrame):
            geo_gdf = gpd.GeoDataFrame(geo_gdf, crs=None)
        geo_gdf.to_file(geojson_path, driver="GeoJSON")
        print(f"  GeoJSON (pixel coords) → {geojson_path.relative_to(repo_root)}")

    # ------------------------------------------------------------------
    # GeoPackage write is handled by text_to_vector.py in the maptools
    # environment — New-MapReader's fiona cannot write GeoPackages reliably.
    # ------------------------------------------------------------------
    if len(geo_gdf) == 0:
        print("\nWarning: no text predictions produced.")
    else:
        print(f"\nInference complete — {len(geo_gdf):,} text instances saved to GeoJSON.")

    print(f"\nTo add the text layer to the GeoPackage, run in a separate terminal:")
    print(f"  conda activate maptools")
    print(f"  python steps/05_vectorise/text/text_to_vector.py --sheet {sheet_id}")
    print(f"\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MapTextPipeline text spotting on patchified map sheet."
    )
    parser.add_argument("--sheet",   required=True,
                        help="Sheet ID")
    parser.add_argument("--weights", default=None,
                        help="Path to .pth weights file "
                             "(default: auto-selects from models/base/MapTextPipeline/)")
    parser.add_argument("--device",  default=None,
                        help="'cuda', 'cpu', or 'default' (auto; default)")
    args = parser.parse_args()
    predict(args.sheet, ROOT, args.weights, args.device)
