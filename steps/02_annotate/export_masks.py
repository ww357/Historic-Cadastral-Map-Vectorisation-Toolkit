"""
Convert labelme JSON annotations to binary mask PNGs, sorted by feature class.

Reads   : data/annotations/labelme_json/<SHEET_ID>/*.json
Writes  : data/annotations/<feature>/<SHEET_ID>/images/*.png  — patch image copy
          data/annotations/<feature>/<SHEET_ID>/masks/*.png   — binary mask
          data/annotations/text/<SHEET_ID>/labels/*.json      — text content sidecars

Feature routing:
  label == "boundary"  → annotations/boundaries/  (linestrip rendered at line_width px)
  label == "water"     → annotations/water/
  label == "building"  → annotations/buildings/
  label == "damage"    → annotations/damage/
  anything else        → annotations/text/  (label value = text content)

Patches annotated with multiple features produce a mask file in each relevant folder.
Patches with no shapes for a given feature are skipped for that feature.

Usage:
    python export_masks.py --sheet SHEET_ID [--line-width N]
"""

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


def render_mask(shapes: list, size: tuple[int, int], line_width: int) -> Image.Image:
    """Render a list of labelme shapes into a binary (L-mode) mask."""
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for shape in shapes:
        pts = [tuple(p) for p in shape["points"]]
        if shape["shape_type"] == "linestrip":
            if len(pts) >= 2:
                draw.line(pts, fill=255, width=line_width)
        elif shape["shape_type"] in ("polygon", "rectangle"):
            if len(pts) >= 3:
                draw.polygon(pts, fill=255)
    return mask


def export_masks(sheet_id: str, line_width_override: int | None):
    cfg     = load_config()
    acfg    = cfg["annotation"]
    ann_dir = ROOT / cfg["paths"]["annotations"]

    known_labels   = set(acfg["feature_labels"])
    label_to_folder = acfg["label_to_folder"]
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

    # Track counts per feature for summary
    counts = defaultdict(int)

    for json_path in json_files:
        with open(json_path) as f:
            data = json.load(f)

        shapes = data.get("shapes", [])
        if not shapes:
            continue

        patch_id   = json_path.stem
        patch_img  = patches_dir / f"{patch_id}.png"
        img_w      = data.get("imageWidth",  512)
        img_h      = data.get("imageHeight", 512)

        # Group shapes by feature label
        feature_shapes = defaultdict(list)
        text_shapes    = []

        for shape in shapes:
            label = shape.get("label", "").strip()
            if label in known_labels:
                feature_shapes[label].append(shape)
            else:
                text_shapes.append(shape)

        # --- Export each feature class ---
        for label, f_shapes in feature_shapes.items():
            folder   = label_to_folder.get(label, label)
            out_base = ann_dir / folder / sheet_id
            img_out  = out_base / "images"
            mask_out = out_base / "masks"
            img_out.mkdir(parents=True, exist_ok=True)
            mask_out.mkdir(parents=True, exist_ok=True)

            mask = render_mask(f_shapes, (img_w, img_h), line_width)
            mask.save(mask_out / f"{patch_id}.png")
            if patch_img.exists():
                shutil.copy(patch_img, img_out / f"{patch_id}.png")

            counts[folder] += 1

        # --- Export text annotations ---
        if text_shapes:
            out_base   = ann_dir / "text" / sheet_id
            img_out    = out_base / "images"
            mask_out   = out_base / "masks"
            label_out  = out_base / "labels"
            for d in (img_out, mask_out, label_out):
                d.mkdir(parents=True, exist_ok=True)

            mask = render_mask(text_shapes, (img_w, img_h), line_width)
            mask.save(mask_out / f"{patch_id}.png")
            if patch_img.exists():
                shutil.copy(patch_img, img_out / f"{patch_id}.png")

            # Sidecar JSON preserving text content and polygon coordinates
            sidecar = {
                "patch_id": patch_id,
                "texts": [
                    {"text": s["label"], "points": s["points"]}
                    for s in text_shapes
                ],
            }
            with open(label_out / f"{patch_id}.json", "w") as f:
                json.dump(sidecar, f, indent=2)

            counts["text"] += 1

    print("Exported:")
    for feature, n in sorted(counts.items()):
        print(f"  {feature:<12} {n} patch(es)")
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
