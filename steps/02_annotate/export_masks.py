"""
Convert labelme JSON annotations to binary mask PNGs, sorted by feature class.

Every unique label drawn in labelme automatically becomes its own feature class —
there is no hardcoded list of allowed labels.  The label name is used directly as
the output folder name (e.g. label "water" → annotations/water/).

One exception: shapes labelled "boundary" (or whatever annotation.boundary_label
is set to in config.yaml) are rendered as linesteps at the configured line width
instead of filled polygons, because boundaries are linestrip annotations.

Reads   : data/annotations/labelme_json/<SHEET_ID>/*.json
Writes  : data/annotations/<label>/<SHEET_ID>/images/*.png  — patch image copy
          data/annotations/<label>/<SHEET_ID>/masks/*.png   — binary mask

Patches annotated with multiple labels produce a mask file in each relevant folder.
Patches with no shapes for a given label are skipped for that label.

Usage:
    python export_masks.py --sheet SHEET_ID [--line-width N]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    p = ROOT / "config.yaml"
    if not p.exists():
        sys.exit(f"config.yaml not found at {p}")
    return yaml.safe_load(p.read_text())


def render_mask(shapes: list, size: tuple[int, int],
                line_width: int, boundary_label: str) -> Image.Image:
    """
    Render a list of labelme shapes into a binary (L-mode) mask.

    Shapes whose label matches boundary_label are drawn as linesteps at
    line_width pixels wide.  All other shapes are filled polygons/rectangles.
    """
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for shape in shapes:
        pts = [tuple(p) for p in shape["points"]]
        if shape.get("label", "") == boundary_label:
            # Boundary annotations: linestrip rendered at line_width
            if shape["shape_type"] == "linestrip" and len(pts) >= 2:
                draw.line(pts, fill=255, width=line_width)
        else:
            # All other labels: filled polygon / rectangle
            if shape["shape_type"] in ("polygon", "rectangle") and len(pts) >= 3:
                draw.polygon(pts, fill=255)
            elif shape["shape_type"] == "linestrip" and len(pts) >= 2:
                # Linestips for non-boundary labels still rendered as lines
                draw.line(pts, fill=255, width=line_width)
    return mask


def export_masks(sheet_id: str, line_width_override: int | None):
    cfg     = load_config()
    acfg    = cfg["annotation"]
    ann_dir = ROOT / cfg["paths"]["annotations"]

    boundary_label = acfg.get("boundary_label", "boundary")
    line_width     = line_width_override or int(acfg["line_width"])

    json_dir    = ann_dir / "labelme_json" / sheet_id
    patches_dir = ROOT / cfg["paths"]["patches"] / "images" / sheet_id

    if not json_dir.exists():
        sys.exit(f"No annotations found at {json_dir}  — run annotate.py first.")
    if not patches_dir.exists():
        sys.exit(f"Patches not found: {patches_dir}")

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        sys.exit(f"No JSON files in {json_dir}")

    print(f"Sheet      : {sheet_id}")
    print(f"JSON files : {len(json_files)}")
    print(f"Line width : {line_width}px\n")

    # counts[label] = number of patches that have at least one shape of that label
    counts: dict[str, int] = defaultdict(int)

    for json_path in json_files:
        with open(json_path) as f:
            data = json.load(f)

        shapes = data.get("shapes", [])
        if not shapes:
            continue

        patch_id  = json_path.stem
        patch_img = patches_dir / f"{patch_id}.png"
        img_w     = data.get("imageWidth",  512)
        img_h     = data.get("imageHeight", 512)

        # Group shapes by their label — every unique label is its own class
        label_shapes: dict[str, list] = defaultdict(list)
        for shape in shapes:
            label = shape.get("label", "").strip()
            if label:
                label_shapes[label].append(shape)

        # Export one mask per unique label found in this patch
        for label, l_shapes in label_shapes.items():
            out_base = ann_dir / label / sheet_id
            img_out  = out_base / "images"
            mask_out = out_base / "masks"
            img_out.mkdir(parents=True, exist_ok=True)
            mask_out.mkdir(parents=True, exist_ok=True)

            mask = render_mask(l_shapes, (img_w, img_h), line_width, boundary_label)
            mask.save(mask_out / f"{patch_id}.png")
            if patch_img.exists():
                shutil.copy(patch_img, img_out / f"{patch_id}.png")

            counts[label] += 1

    if not counts:
        print("No annotated shapes found.")
        return

    # Summary: unique classes actually present in the annotations
    print(f"Unique classes found: {len(counts)}")
    print("Exported:")
    for label, n in sorted(counts.items()):
        print(f"  {label:<16} {n} patch(es)")
    print(f"\nAnnotations written to {ann_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export labelme annotations to binary masks sorted by feature."
    )
    parser.add_argument("--sheet",      required=True, help="Sheet ID")
    parser.add_argument("--line-width", type=int, default=None,
                        help="Override line width for linestrip rendering (default from config)")
    args = parser.parse_args()
    export_masks(args.sheet, args.line_width)
