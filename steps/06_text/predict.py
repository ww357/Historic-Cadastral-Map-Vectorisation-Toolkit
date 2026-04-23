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
    python predict.py --sheet SHEET_ID [--weights path/to/weights.pth] [--device cpu]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
import yaml

ROOT = Path(__file__).resolve().parents[2]


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

    # Settings — CLI args override config.yaml
    weights_path  = resolve_weights(weights_arg, repo_root, paths_cfg)
    device        = device_arg or text_cfg.get("device", "default")
    min_ioa       = float(text_cfg.get("min_ioa", 0.7))
    cfg_file_path = repo_root / text_cfg.get(
        "cfg_file",
        "models/MapTextPipeline/configs/ViTAEv2_S/rumsey/final_rumsey.yaml",
    )

    print(f"Sheet    : {sheet_id}")
    print(f"Weights  : {weights_path.relative_to(repo_root)}")
    print(f"Config   : {cfg_file_path.relative_to(repo_root)}")
    print(f"Device   : {device}")
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
    # Instantiate runner
    # ------------------------------------------------------------------
    runner = MapTextRunner(
        patch_df,
        parent_df,
        cfg_file     = str(cfg_file_path),
        weights_file = str(weights_path),
        device       = device,
    )

    # ------------------------------------------------------------------
    # Run detection + recognition on all patches
    # ------------------------------------------------------------------
    print("Running text spotting on all patches...")
    runner.run_all(min_ioa=min_ioa)
    # patch_predictions is a dict keyed by basename; patch_preds does not exist
    preds_dict = getattr(runner, "patch_predictions", None) or {}
    n_patch_preds = sum(len(v) for v in preds_dict.values()) if preds_dict else 0
    print(f"  Patch-level instances: {n_patch_preds}")

    # ------------------------------------------------------------------
    # Scale up to parent image + cross-patch deduplication
    # ------------------------------------------------------------------
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
    geojson_path = pred_dir / "text_preds.geojson"

    if has_georef:
        print("Georeferencing predictions...")
        # to_geojson() handles CRS conversion internally and saves to WGS84.
        # We then read it back with geopandas to get a clean GeoDataFrame —
        # this avoids the fiona NULL pointer error that occurs when
        # convert_to_coords() returns a GeoDataFrame without a CRS set.
        runner.to_geojson(str(geojson_path))
        print(f"  GeoJSON → {geojson_path.relative_to(repo_root)}")

        geo_gdf = gpd.read_file(geojson_path)
        print(f"  Georeferenced instances: {len(geo_gdf)}")
    else:
        # No CRS — use parent pixel bounds directly
        geo_gdf = parent_preds_df.copy()
        if not isinstance(geo_gdf, gpd.GeoDataFrame):
            geo_gdf = gpd.GeoDataFrame(geo_gdf, crs=None)
        geo_gdf.to_file(geojson_path, driver="GeoJSON")
        print(f"  GeoJSON (pixel coords) → {geojson_path.relative_to(repo_root)}")

    # pandas 2.x / Python 3.12: StringDtype columns cause fiona to crash
    # with a TypeError in infer_schema — cast to plain object dtype.
    str_cols = geo_gdf.select_dtypes(include="string").columns.tolist()
    if str_cols:
        geo_gdf[str_cols] = geo_gdf[str_cols].astype(object)

    # ------------------------------------------------------------------
    # GeoPackage write is handled by fix_text_layer.py in the maptools
    # environment. The New-MapReader fiona build cannot write GeoPackages
    # reliably (NULL pointer error), whereas maptools fiona can.
    # ------------------------------------------------------------------
    if len(geo_gdf) == 0:
        print("\nWarning: no text predictions produced.")
    else:
        print(f"\nInference complete — {len(geo_gdf):,} text instances saved to GeoJSON.")

    print(f"\nTo add the text layer to the GeoPackage, run in a separate terminal:")
    print(f"  conda activate maptools")
    print(f"  python fix_text_layer.py --sheet {sheet_id}")
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
