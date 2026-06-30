"""
Diagnostic visualisation for point-prompted SAM parcel segmentation.

For a sample of tiles, re-runs SAM inference and saves a PNG panel showing:

  Col 0  : raw tile image
  Col 1  : tile + prompt points (green = foreground, red = background)
  Col 2-4: all 3 SAM mask candidates, score printed, winner highlighted

Output is written to:
    data/predictions/parcels/<SHEET>/viz/

Usage:
    conda activate polygons
    python steps/04_predict/parcels/parcel_visualise.py --sheet Timberscombe
    python steps/04_predict/parcels/parcel_visualise.py --sheet Timberscombe \\
        --n-tiles 20 --seed 99 --points-per-tile 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml

ROOT       = Path(__file__).resolve().parents[3]
MAPSAM_DIR = ROOT / "models" / "MapSAM"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MAPSAM_DIR))

# ── PROJ / pyogrio compat (mirror of parcel_predict.py) ──────────────────────
if "PROJ_DATA" not in os.environ:
    _env_root = Path(sys.executable).parents[1]
    _candidates = [_env_root / "share" / "proj"]
    import importlib.util as _ilu
    _s = _ilu.find_spec("pyproj")
    if _s and _s.submodule_search_locations:
        _p = Path(list(_s.submodule_search_locations)[0])
        _candidates += [_p / "proj_dir" / "share" / "proj", _p / "data"]
    for _c in _candidates:
        if (_c / "proj.db").exists():
            os.environ["PROJ_DATA"] = str(_c)
            break

try:
    import rasterio
    import rasterio.windows
    from rasterio.transform import Affine
except ImportError:
    sys.exit("rasterio not installed — run: conda activate polygons")

import pandas as pd                                        # noqa: E402
import sqlite3, struct                                      # noqa: E402
import torch                                               # noqa: E402
from segment_anything import SamPredictor, sam_model_registry  # noqa: E402

# ── Reuse helpers from parcel_predict ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from parcel_predict import (                               # noqa: E402
    load_config,
    read_tile_rgb,
    read_gpkg_points_wkb,
    set_image_safe,
)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def overlay_mask(img_rgb: np.ndarray, mask: np.ndarray,
                 colour: tuple[float, float, float] = (0.2, 0.6, 1.0),
                 alpha: float = 0.45) -> np.ndarray:
    """Return a copy of img_rgb with mask blended in at alpha transparency."""
    out = img_rgb.astype(np.float32) / 255.0
    for c, v in enumerate(colour):
        out[..., c] = np.where(mask > 0,
                               out[..., c] * (1 - alpha) + v * alpha,
                               out[..., c])
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def draw_points(img_rgb: np.ndarray,
                fg_coords: list[tuple[float, float]],
                bg_coords: list[tuple[float, float]],
                radius: int = 8) -> np.ndarray:
    """Draw foreground (green) and background (red) prompt dots on image."""
    out = img_rgb.copy()
    for x, y in bg_coords:
        cv2.circle(out, (int(x), int(y)), radius + 2, (0, 0, 0), -1)
        cv2.circle(out, (int(x), int(y)), radius, (220, 50, 50), -1)
    for x, y in fg_coords:
        cv2.circle(out, (int(x), int(y)), radius + 2, (0, 0, 0), -1)
        cv2.circle(out, (int(x), int(y)), radius, (50, 205, 50), -1)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise SAM parcel predictions tile-by-tile."
    )
    parser.add_argument("--sheet",           required=True)
    parser.add_argument("--n-tiles",         type=int, default=12,
                        help="Number of tiles to visualise (default 12)")
    parser.add_argument("--points-per-tile", type=int, default=3,
                        help="Max points to visualise per tile (default 3)")
    parser.add_argument("--seed",            type=int, default=42,
                        help="Random seed for tile sampling (default 42)")
    parser.add_argument("--device",          default=None)
    args = parser.parse_args()
    sheet_id = args.sheet

    rng = np.random.default_rng(args.seed)

    cfg   = load_config()
    paths = cfg["paths"]
    pcfg  = cfg.get("parcels", {})

    tile_size   = int(pcfg.get("tile_size",      1024))
    sam_size    = int(pcfg.get("sam_input_size",  1024))
    overlap     = int(pcfg.get("overlap",          128))
    points_file = pcfg.get("points_file", "Holnicote Apportionment Points.gpkg")
    max_neg_pts = int(pcfg.get("max_neg_points",     6))
    min_cov     = float(cfg.get("patchify", {}).get("min_mask_coverage", 0.01))

    stride  = tile_size - overlap
    scale   = sam_size / tile_size
    half_ov = overlap // 2

    tif_path    = ROOT / paths["raw"]          / sheet_id / f"{sheet_id}.tif"
    points_path = ROOT / paths["parcel_points"] / points_file
    out_dir     = ROOT / paths["predictions"]  / "parcels" / sheet_id / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    for p, lbl in [(tif_path, "TIF"), (points_path, "points GeoPackage")]:
        if not p.exists():
            sys.exit(f"{lbl} not found: {p}")

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device : {device}")

    # ── SAM ───────────────────────────────────────────────────────────────────
    sam_ckpt = (ROOT / paths["models_base"]
                / "MapSAM" / "origional_weights" / "sam_vit_b_01ec64.pth")
    if not sam_ckpt.exists():
        sys.exit(f"SAM weights not found: {sam_ckpt}")

    sam, _ = sam_model_registry["vit_b"](
        image_size=sam_size, num_classes=4, checkpoint=str(sam_ckpt)
    )
    sam.eval().to(device)
    predictor = SamPredictor(sam)
    print(f"SAM    : {sam_ckpt.name}")

    # ── TIF metadata ──────────────────────────────────────────────────────────
    with rasterio.open(tif_path) as src:
        tif_tf     = src.transform
        tif_w      = src.width
        tif_h      = src.height
        tif_bounds = src.bounds

    # ── Mask ──────────────────────────────────────────────────────────────────
    mask_img = None
    for _mp in [ROOT / paths["masks"] / sheet_id / f"{sheet_id}.png",
                ROOT / paths["masks"] / f"{sheet_id}.png"]:
        if _mp.exists():
            _raw = cv2.imread(str(_mp), cv2.IMREAD_GRAYSCALE)
            if _raw is not None:
                if (_raw.shape[1], _raw.shape[0]) != (tif_w, tif_h):
                    _raw = cv2.resize(_raw, (tif_w, tif_h),
                                      interpolation=cv2.INTER_NEAREST)
                mask_img = _raw
                print(f"Mask   : {_mp.name}")
                break

    # ── Parcel points ─────────────────────────────────────────────────────────
    print("Reading parcel points ...")
    all_pts = read_gpkg_points_wkb(points_path)
    pts_in  = all_pts[
        (all_pts["_geom_x"] >= tif_bounds.left)  &
        (all_pts["_geom_x"] <= tif_bounds.right) &
        (all_pts["_geom_y"] >= tif_bounds.bottom)&
        (all_pts["_geom_y"] <= tif_bounds.top)
    ].copy().reset_index(drop=True)
    pts_in["_px_col"] = (pts_in["_geom_x"] - tif_tf.c) / tif_tf.a
    pts_in["_px_row"] = (pts_in["_geom_y"] - tif_tf.f) / tif_tf.e
    print(f"Points : {len(pts_in)} within sheet")

    # ── Build list of candidate tiles ─────────────────────────────────────────
    col_cores = list(range(0, tif_w, stride))
    row_cores = list(range(0, tif_h, stride))

    candidate_tiles = []
    for col_core in col_cores:
        for row_core in row_cores:
            col_core_end = min(col_core + stride, tif_w)
            row_core_end = min(row_core + stride, tif_h)

            # Mask coverage
            if mask_img is not None:
                cov = float((mask_img[row_core:row_core_end,
                                      col_core:col_core_end] > 0).mean())
                if cov < min_cov:
                    continue

            # Must have at least one point
            tile_pts = pts_in[
                (pts_in["_px_col"] >= col_core) &
                (pts_in["_px_col"] <  col_core_end) &
                (pts_in["_px_row"] >= row_core) &
                (pts_in["_px_row"] <  row_core_end)
            ]
            if len(tile_pts) == 0:
                continue

            candidate_tiles.append((col_core, row_core, col_core_end,
                                    row_core_end, tile_pts.copy()))

    print(f"Tiles  : {len(candidate_tiles)} eligible, sampling {args.n_tiles}")
    if len(candidate_tiles) == 0:
        sys.exit("No eligible tiles found — check mask and points.")

    chosen = rng.choice(len(candidate_tiles),
                        size=min(args.n_tiles, len(candidate_tiles)),
                        replace=False)

    # ── Per-tile visualisation ────────────────────────────────────────────────
    MASK_COLOURS = [
        (0.20, 0.60, 1.00),   # blue   — mask 0
        (1.00, 0.60, 0.10),   # orange — mask 1
        (0.20, 0.85, 0.40),   # green  — mask 2
    ]

    with rasterio.open(tif_path) as src:
        for tile_no, tile_idx in enumerate(chosen, 1):
            col_core, row_core, col_core_end, row_core_end, tile_pts = \
                candidate_tiles[tile_idx]

            col_read = max(0, col_core - half_ov)
            row_read = max(0, row_core - half_ov)

            tile_rgb = read_tile_rgb(src, col_read, row_read, tile_size)
            tile_sam = tile_rgb if tile_size == sam_size else cv2.resize(
                tile_rgb, (sam_size, sam_size), interpolation=cv2.INTER_AREA)
            tile_sam = np.ascontiguousarray(tile_sam, dtype=np.uint8)

            set_image_safe(predictor, tile_sam)
            if isinstance(predictor.features, tuple):
                predictor.features = predictor.features[0]

            # Pick up to --points-per-tile points to visualise
            pt_rows = tile_pts.sample(
                n=min(args.points_per_tile, len(tile_pts)),
                random_state=args.seed
            )

            n_pts  = len(pt_rows)
            n_cols  = 5    # raw | prompts | mask0 | mask1 | mask2
            fig, axes = plt.subplots(
                n_pts, n_cols,
                figsize=(n_cols * 3.5, n_pts * 3.5 + 0.6),
                squeeze=False,
            )
            fig.suptitle(
                f"{sheet_id}  |  tile ({col_core},{row_core})  "
                f"[{col_core_end-col_core}×{row_core_end-row_core} px core]",
                fontsize=10, fontweight="bold"
            )

            col_headers = ["Raw tile", "Prompts",
                           "Mask 0", "Mask 1", "Mask 2"]
            for ci, hdr in enumerate(col_headers):
                axes[0, ci].set_title(hdr, fontsize=8)

            for row_i, (pt_idx, pt) in enumerate(pt_rows.iterrows()):
                # ── Prompt coords ──────────────────────────────────────────
                fg_sam = [(float((pt["_px_col"] - col_read) * scale),
                           float((pt["_px_row"] - row_read) * scale))]

                neg_coords_sam: list[tuple[float, float]] = []
                if max_neg_pts > 0:
                    others = tile_pts[tile_pts.index != pt_idx]
                    if len(others) > 0:
                        dx = others["_px_col"].values - float(pt["_px_col"])
                        dy = others["_px_row"].values - float(pt["_px_row"])
                        order = np.argsort(dx*dx + dy*dy)
                        near  = others.iloc[order[:max_neg_pts]]
                        for _, np_pt in near.iterrows():
                            nx = float((np_pt["_px_col"] - col_read) * scale)
                            ny = float((np_pt["_px_row"] - row_read) * scale)
                            nx = float(np.clip(nx, 0, sam_size - 1))
                            ny = float(np.clip(ny, 0, sam_size - 1))
                            neg_coords_sam.append((nx, ny))

                all_coords = np.array(fg_sam + neg_coords_sam)
                all_labels = np.array([1] + [0] * len(neg_coords_sam),
                                      dtype=np.int32)

                # ── SAM inference ──────────────────────────────────────────
                try:
                    masks, scores, _ = predictor.predict(
                        point_coords=all_coords,
                        point_labels=all_labels,
                        multimask_output=True,
                    )
                    # Force fresh NumPy-2.0-native arrays — torch-backed arrays
                    # from .numpy() cause numpy.where / np.asarray to fail.
                    masks  = [np.array(m, dtype=np.uint8) for m in masks]
                    scores = [float(s) for s in scores]
                    best_idx = int(np.argmax(scores))
                    ok = True
                except Exception as exc:
                    ok = False
                    scores = [0.0, 0.0, 0.0]
                    masks  = [np.zeros((sam_size, sam_size), dtype=np.uint8)] * 3
                    best_idx = 0
                    print(f"  SAM error: {exc}")

                # ── Col 0: raw tile ────────────────────────────────────────
                ax = axes[row_i, 0]
                ax.imshow(tile_sam)
                ax.set_ylabel(
                    f"rowid={pt.get('rowid','?')}\n"
                    f"ParcelID={pt.get('ParcelID','?')}",
                    fontsize=7, rotation=0, labelpad=60, va="center"
                )
                ax.axis("off")

                # ── Col 1: tile + prompts ──────────────────────────────────
                prompt_img = draw_points(tile_sam, fg_sam, neg_coords_sam,
                                         radius=max(4, sam_size // 120))
                axes[row_i, 1].imshow(prompt_img)
                axes[row_i, 1].axis("off")

                # ── Cols 2-4: mask candidates ──────────────────────────────
                for mi in range(3):
                    ax = axes[row_i, 2 + mi]
                    masked_img = overlay_mask(tile_sam, masks[mi],
                                              colour=MASK_COLOURS[mi])
                    masked_img = draw_points(masked_img, fg_sam, neg_coords_sam,
                                             radius=max(3, sam_size // 150))
                    ax.imshow(masked_img)

                    border_col = "gold" if mi == best_idx else "none"
                    for spine in ax.spines.values():
                        spine.set_edgecolor(border_col)
                        spine.set_linewidth(3 if mi == best_idx else 1)

                    status = "★ SELECTED" if mi == best_idx else ""
                    score_txt = f"score: {scores[mi]:.3f}  {status}"
                    area_px = int(masks[mi].sum())
                    ax.set_xlabel(
                        f"{score_txt}\narea: {area_px:,} px",
                        fontsize=7,
                        color="goldenrod" if mi == best_idx else "grey"
                    )
                    ax.axis("off")

            # ── Legend ────────────────────────────────────────────────────
            legend_patches = [
                mpatches.Patch(color=(50/255, 205/255, 50/255),
                               label="Foreground point"),
                mpatches.Patch(color=(220/255, 50/255, 50/255),
                               label="Background point"),
                mpatches.Patch(color="gold", label="Selected mask (★)"),
            ]
            fig.legend(handles=legend_patches, loc="lower center",
                       ncol=3, fontsize=8, framealpha=0.9,
                       bbox_to_anchor=(0.5, 0.0))

            plt.tight_layout(rect=[0, 0.04, 1, 1])

            out_png = out_dir / f"tile_{tile_no:03d}_c{col_core}_r{row_core}.png"
            fig.savefig(str(out_png), dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"  [{tile_no}/{len(chosen)}] {out_png.name}")

    print(f"\nDone — {len(chosen)} panels saved to:\n  {out_dir}")


if __name__ == "__main__":
    main()
