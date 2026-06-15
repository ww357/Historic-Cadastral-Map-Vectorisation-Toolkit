"""
Visual sanity-check for parcel SAM training inputs.

Renders what the model ACTUALLY sees per step, straight out of
ParcelAnnotationDataset: the (augmented, resized) patch, the GT mask it is
asked to reproduce, and the simulated point prompts —
  • green dot  = foreground prompt ("segment the parcel at this point")
  • red dots   = background prompts ("not these neighbouring parcels")

Use this to confirm qualitatively that:
  • the foreground point lands INSIDE the annotated parcel
  • the GT mask aligns with the parcel in the image
  • augmentation hasn't rotated/zoomed the parcel out of frame or emptied the mask
  • background points sit in clear background, not on the target parcel

This visualises the SAME tensors the training loop consumes, so if a prompt
or mask looks wrong here, the model is being trained on that wrong signal.

Usage:
    conda activate polygons
    python steps/03_finetune/parcels/visualise_training_samples.py --all-sheets -n 16
    python steps/03_finetune/parcels/visualise_training_samples.py --sheet Porlock -n 12 --no-aug

Output:
    data/predictions/parcels/_training_check/<timestamp>/*.png
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))

# Reuse the exact dataset + helpers the trainer uses, so this reflects reality.
from train import ParcelAnnotationDataset   # noqa: E402


def gather_pairs(sheet_ids: list[str], base_ann: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for sheet in sheet_ids:
        imgs_dir = base_ann / sheet / "images"
        msks_dir = base_ann / sheet / "masks"
        if not imgs_dir.exists():
            print(f"  [{sheet}] no exported masks at {imgs_dir} — skipping")
            continue
        sheet_pairs = sorted(
            (imgs_dir / p.name, msks_dir / p.name)
            for p in imgs_dir.glob("*.png")
            if (msks_dir / p.name).exists()
        )
        print(f"  [{sheet}] {len(sheet_pairs)} annotated parcels")
        pairs.extend(sheet_pairs)
    return pairs


def render_sample(sample: dict, img_size: int) -> np.ndarray:
    """Build an RGB panel: patch with GT mask overlay + prompt points."""
    # Image tensor is (3, H, W) float in [0, 1].
    # Convert inside torch, then clone the .numpy() result into a fresh
    # NumPy-native array — torch's .numpy() alone yields an array cv2 / NumPy
    # 2.x reject, and passing the tensor straight to np.array() trips torch's
    # broken __array__(dtype, copy). .numpy() → np.array(..., dtype=...) works.
    img_t = sample["image"].permute(1, 2, 0).mul(255).round().clamp(0, 255).to(torch.uint8)
    img   = np.array(img_t.cpu().numpy(), dtype=np.uint8).copy()

    # Low-res GT mask (img_size // 4) → upscale to img_size for overlay
    low = np.array(sample["mask_low"].cpu().numpy(), dtype=np.float32)
    gt  = cv2.resize(low, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

    # Semi-transparent cyan fill over the GT mask
    overlay = img.copy()
    overlay[gt > 0.5] = (0, 200, 255)
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

    # GT contour for a crisp boundary
    cont, _ = cv2.findContours((gt > 0.5).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cont, -1, (0, 255, 255), 2)

    # Prompt points — coords are in img_size pixel space (col, row)
    coords = np.array(sample["coords"].cpu().numpy(), dtype=np.float32)
    labels = np.array(sample["labels"].cpu().numpy(), dtype=np.int32)
    for (cx, cy), lab in zip(coords, labels):
        cx, cy = int(round(cx)), int(round(cy))
        if lab == 1:   # foreground
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), -1)
            cv2.circle(img, (cx, cy), 11, (0, 0, 0), 2)
        else:          # background
            cv2.circle(img, (cx, cy), 7, (255, 40, 40), -1)
            cv2.circle(img, (cx, cy), 8, (0, 0, 0), 1)

    # Label banner
    fg_inside = bool(gt[min(int(coords[0][1]), img_size - 1),
                        min(int(coords[0][0]), img_size - 1)] > 0.5)
    tag = f"{sample['case_name']}   fg_in_mask={'YES' if fg_inside else 'NO!'}"
    colour = (0, 220, 0) if fg_inside else (0, 0, 255)
    cv2.rectangle(img, (0, 0), (img_size, 34), (0, 0, 0), -1)
    cv2.putText(img, tag, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)   # cv2.imwrite expects BGR


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualise parcel SAM training inputs.")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--sheet", nargs="+", metavar="SHEET")
    grp.add_argument("--all-sheets", action="store_true")
    ap.add_argument("-n", "--num", type=int, default=16, help="samples to render")
    ap.add_argument("--no-aug", action="store_true",
                    help="render without augmentation (raw annotations)")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg   = yaml.safe_load((ROOT / args.config).read_text())
    paths = cfg["paths"]
    pcfg  = cfg.get("parcels", {})
    ftcfg = cfg.get("parcel_finetune", {})

    img_size = int(pcfg.get("sam_input_size", 1024))
    # Mirror the trainer's negative-prompt settings (NOT inference's), so the
    # check shows exactly the prompts the model will be trained on.
    n_bg     = int(ftcfg.get("neg_points", 12))
    ring_px  = int(ftcfg.get("neg_ring_px", 48))
    min_zoom = int(ftcfg.get("min_zoom_area_px", 1500))

    base_ann = ROOT / paths["annotations"] / "parcel"
    if args.all_sheets:
        sheet_ids = sorted(d.name for d in base_ann.iterdir() if d.is_dir()) \
                    if base_ann.exists() else []
        if not sheet_ids:
            sys.exit(f"No exported parcel masks under {base_ann}. "
                     "Run export_masks.py first.")
    else:
        sheet_ids = args.sheet
    print(f"Sheets   : {', '.join(sheet_ids)}")

    pairs = gather_pairs(sheet_ids, base_ann)
    if not pairs:
        sys.exit("No image+mask pairs found.")

    ds = ParcelAnnotationDataset(
        pairs, img_size, n_bg=n_bg,
        do_aug=not args.no_aug, min_zoom_area_px=min_zoom,
        neg_ring_px=ring_px,
    )
    n = min(args.num, len(ds))

    out_dir = (ROOT / paths["predictions"] / "parcels" / "_training_check"
               / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Spread the sample indices across the dataset rather than just the first n
    idxs = np.linspace(0, len(ds) - 1, n).round().astype(int)
    bad  = 0
    for k, i in enumerate(idxs):
        sample = ds[int(i)]
        panel  = render_sample(sample, img_size)
        coords = np.array(sample["coords"].cpu().numpy(), dtype=np.float32)
        low    = np.array(sample["mask_low"].cpu().numpy(), dtype=np.float32)
        gt     = cv2.resize(low, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        if gt[min(int(coords[0][1]), img_size - 1),
              min(int(coords[0][0]), img_size - 1)] <= 0.5:
            bad += 1
        cv2.imwrite(str(out_dir / f"{k:02d}_{sample['case_name']}.png"), panel)

    print(f"\nWrote {n} panels → {out_dir.relative_to(ROOT)}")
    print(f"Aug      : {'OFF' if args.no_aug else 'ON'}")
    if bad:
        print(f"  ⚠  {bad}/{n} samples have the foreground point OUTSIDE the GT "
              f"mask — those are bad training signals.")
    else:
        print("  ✓ all foreground prompts land inside their GT mask.")


if __name__ == "__main__":
    main()
