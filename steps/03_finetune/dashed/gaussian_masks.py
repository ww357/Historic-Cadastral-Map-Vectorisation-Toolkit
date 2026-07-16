"""
Convert labelme "dashed" linestrip annotations into Gaussian heatmap training
targets for the DashedLineUNet (see models/DashedLineUNet/).

This reads labelme JSON directly rather than the binary masks produced by
steps/02_annotate/export_masks.py, because the training target here is a
continuous distance-transform heatmap (which is what lets the model bridge
small ink gaps between dashes), not a binary mask. Run this INSTEAD of (not
in addition to, though it's harmless either way) export_masks.py for the
"dashed" label — export_masks.py is still needed for any other labels
(boundary, water, ...) drawn in the same sheet.

Reads (per sheet):
    data/patches/images/<SHEET_ID>/*.png                 — 512px patches (step 01)
    data/annotations/labelme_json/<SHEET_ID>/*.json      — labelme output (step 02)

For every patch that has a saved JSON (i.e. was reviewed by the annotator,
whether or not they drew a "dashed" shape), writes:
    data/annotations/dashed/<SHEET_ID>/masks/<patch_id>.png      — binary mask
                                                                    (also makes Patch_Grid
                                                                    pick up ann_dashed, same
                                                                    convention as other labels)
    data/annotations/dashed/<SHEET_ID>/gaussian/<patch_id>.npy   — float32 heatmap, the
                                                                    actual training target
    data/annotations/dashed/<SHEET_ID>/gaussian/<patch_id>_preview.png  — visual check

Patches whose JSON has no "dashed" shape are written as all-zero (mask and
heatmap) — a confirmed-negative example, NOT an unreviewed one, since a JSON
only exists if the annotator opened and saved that patch. This is what lets
the model learn to suppress lines it shouldn't detect, and mirrors how the
original (pre-integration) model was trained. Patches with no JSON at all
were never reviewed and are correctly skipped entirely.

Usage:
    python steps/03_finetune/dashed/gaussian_masks.py [--sheet SHEET_ID]
    (omit --sheet to (re)process every sheet under data/annotations/labelme_json/)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[3]

LABEL = "dashed"


def load_config() -> dict:
    p = ROOT / "config.yaml"
    if not p.exists():
        sys.exit(f"config.yaml not found at {p}")
    return yaml.safe_load(p.read_text())


def shapes_to_binary_mask(shapes: list, h: int, w: int, line_width: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for shape in shapes:
        if shape.get("label", "") != LABEL:
            continue
        if shape["shape_type"] not in ("linestrip", "line"):
            continue
        pts = np.array(shape["points"], dtype=np.float32).reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=line_width)
    return mask


def binary_mask_to_heatmap(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Euclidean distance transform -> Gaussian heatmap, float32 in [0,1]."""
    inverted = (mask == 0).astype(np.uint8)   # 0 on the line, 1 elsewhere
    edt = distance_transform_edt(inverted)
    heatmap = np.exp(-(edt ** 2) / (2 * sigma ** 2))
    return heatmap.astype(np.float32)


def process_sheet(sheet_id: str, cfg: dict) -> int:
    line_width = int(cfg["annotation"]["line_width"])
    sigma      = float(cfg["dashed"]["sigma"])

    json_dir = ROOT / cfg["paths"]["annotations"] / "labelme_json" / sheet_id
    out_base = ROOT / cfg["paths"]["annotations"] / LABEL / sheet_id
    mask_out = out_base / "masks"
    heat_out = out_base / "gaussian"

    if not json_dir.exists():
        print(f"  {sheet_id}: no labelme JSON found at {json_dir} — skipping")
        return 0

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"  {sheet_id}: no annotated patches — skipping")
        return 0

    mask_out.mkdir(parents=True, exist_ok=True)
    heat_out.mkdir(parents=True, exist_ok=True)

    n_positive = 0
    for json_path in json_files:
        with open(json_path) as f:
            data = json.load(f)

        patch_id = json_path.stem
        h = data.get("imageHeight", 512)
        w = data.get("imageWidth", 512)
        shapes = [s for s in data.get("shapes", []) if s.get("label", "") == LABEL]

        binary  = shapes_to_binary_mask(shapes, h, w, line_width)
        heatmap = binary_mask_to_heatmap(binary, sigma)

        cv2.imwrite(str(mask_out / f"{patch_id}.png"), binary)
        np.save(heat_out / f"{patch_id}.npy", heatmap)
        cv2.imwrite(str(heat_out / f"{patch_id}_preview.png"), (heatmap * 255).astype(np.uint8))

        if shapes:
            n_positive += 1

    print(f"  {sheet_id}: {len(json_files)} reviewed patch(es) "
          f"({n_positive} with a dashed line, {len(json_files) - n_positive} confirmed negative)")
    return len(json_files)


def main():
    parser = argparse.ArgumentParser(
        description='Build Gaussian heatmap training targets for the "dashed" label.'
    )
    parser.add_argument("--sheet", default=None,
                        help="Restrict to one sheet (default: process every sheet with "
                             "labelme annotations)")
    args = parser.parse_args()

    cfg = load_config()

    if args.sheet:
        sheets = [args.sheet]
    else:
        labelme_root = ROOT / cfg["paths"]["annotations"] / "labelme_json"
        if not labelme_root.exists():
            sys.exit(f"No annotations found at {labelme_root} — run annotate.py first.")
        sheets = sorted(p.name for p in labelme_root.iterdir() if p.is_dir())

    if not sheets:
        sys.exit("No sheets to process.")

    print(f"Processing {len(sheets)} sheet(s) for label '{LABEL}':")
    total = sum(process_sheet(sheet_id, cfg) for sheet_id in sheets)

    print(f"\nDone. {total} patch(es) written across {len(sheets)} sheet(s).")
    print(f"  -> {ROOT / cfg['paths']['annotations'] / LABEL}/<sheet>/masks/")
    print(f"  -> {ROOT / cfg['paths']['annotations'] / LABEL}/<sheet>/gaussian/")


if __name__ == "__main__":
    main()
