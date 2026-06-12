"""
Create 1024px annotation patches for SAM parcel fine-tuning.

Tiles the raw sheet TIF at 1024px (matching the SAM inference tile size) and
saves each in-mask patch as a PNG ready for labelme annotation.  The user draws
polygon shapes labelled "parcel" on the patches in labelme; labelme saves a JSON
file alongside each PNG.  After annotation run export_masks.py with --patches-dir
and --json-dir pointing at this directory to extract binary masks.

Output
------
    data/patches/parcel/<SHEET_ID>/<patch_id>.png   ← annotate these with labelme

Usage
-----
    conda activate maptools
    python steps/01_patchify/parcels/parcel_patchify.py --sheet Timberscombe

Then annotate:
    python steps/02_annotate/annotate_parcels.py --sheet Timberscombe

Then export masks:
    python steps/02_annotate/export_masks.py --sheet Timberscombe \\
        --patches-dir data/patches/parcel/Timberscombe \\
        --json-dir    data/patches/parcel/Timberscombe

Then fine-tune:
    python steps/03_finetune/parcels/train.py --sheet Timberscombe
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]

# ── PROJ / rasterio compat ────────────────────────────────────────────────────
if "PROJ_DATA" not in os.environ:
    _env_root = Path(sys.executable).parents[1]
    _cands = [_env_root / "share" / "proj"]
    import importlib.util as _ilu
    _s = _ilu.find_spec("pyproj")
    if _s and _s.submodule_search_locations:
        _p = Path(list(_s.submodule_search_locations)[0])
        _cands += [_p / "proj_dir" / "share" / "proj", _p / "data"]
    for _c in _cands:
        if (_c / "proj.db").exists():
            os.environ["PROJ_DATA"] = str(_c)
            break

try:
    import rasterio
    import rasterio.windows
except ImportError:
    sys.exit("rasterio required — conda activate maptools (or polygons)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create 1024px annotation patches for SAM parcel fine-tuning."
    )
    parser.add_argument("--sheet",    required=True,
                        help="Sheet ID — must match a subfolder in data/raw/")
    parser.add_argument("--overlap",  type=int, default=None,
                        help="Override overlap between patches (default from config)")
    args = parser.parse_args()
    sheet_id = args.sheet

    cfg   = yaml.safe_load((ROOT / "config.yaml").read_text())
    paths = cfg["paths"]
    pcfg  = cfg.get("parcels", {})
    ptcfg = cfg.get("patchify", {})

    patch_size  = int(pcfg.get("sam_input_size", 1024))   # 1024px to match inference
    overlap     = args.overlap if args.overlap is not None else int(pcfg.get("overlap", 128))
    min_cov     = float(ptcfg.get("min_mask_coverage", 0.01))
    stride      = patch_size - overlap
    pad_val     = int(ptcfg.get("pad_value", 255))

    tif_path = ROOT / paths["raw"] / sheet_id / f"{sheet_id}.tif"
    out_dir  = ROOT / "data" / "patches" / "parcel" / sheet_id

    if not tif_path.exists():
        sys.exit(f"TIF not found: {tif_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load mask ─────────────────────────────────────────────────────────────
    mask_img = None
    for _mp in [ROOT / paths["masks"] / sheet_id / f"{sheet_id}.png",
                ROOT / paths["masks"] / f"{sheet_id}.png"]:
        if _mp.exists():
            _raw = cv2.imread(str(_mp), cv2.IMREAD_GRAYSCALE)
            if _raw is not None:
                mask_img = _raw
                print(f"Mask      : {_mp.name}")
                break
    if mask_img is None:
        print("Mask      : not found — exporting all patches (no coverage filter)")

    # ── TIF metadata ──────────────────────────────────────────────────────────
    with rasterio.open(tif_path) as src:
        tif_w   = src.width
        tif_h   = src.height
        n_bands = src.count

    if mask_img is not None and (mask_img.shape[1] != tif_w or mask_img.shape[0] != tif_h):
        mask_img = cv2.resize(mask_img, (tif_w, tif_h), interpolation=cv2.INTER_NEAREST)

    col_starts = list(range(0, tif_w, stride))
    row_starts = list(range(0, tif_h, stride))
    n_total    = len(col_starts) * len(row_starts)

    print(f"Sheet     : {sheet_id}  ({tif_w}×{tif_h} px)")
    print(f"Patch     : {patch_size}px  overlap={overlap}px  stride={stride}px")
    print(f"Grid      : {len(col_starts)}×{len(row_starts)} = {n_total} candidates")

    saved = skipped = 0

    with rasterio.open(tif_path) as src:
        for row_start in row_starts:
            for col_start in col_starts:
                row_end = min(row_start + patch_size, tif_h)
                col_end = min(col_start + patch_size, tif_w)

                # ── Mask coverage filter ──────────────────────────────────────
                if mask_img is not None:
                    cov = float((mask_img[row_start:row_end,
                                          col_start:col_end] > 0).mean())
                    if cov < min_cov:
                        skipped += 1
                        continue

                # ── Read patch ────────────────────────────────────────────────
                actual_w = col_end - col_start
                actual_h = row_end - row_start
                patch    = np.full((patch_size, patch_size, 3), pad_val, dtype=np.uint8)

                if actual_w > 0 and actual_h > 0:
                    win  = rasterio.windows.Window(col_start, row_start,
                                                   actual_w, actual_h)
                    if n_bands >= 3:
                        data = src.read([1, 2, 3], window=win)
                    else:
                        grey = src.read(1, window=win)
                        data = np.stack([grey, grey, grey], axis=0)
                    patch[:actual_h, :actual_w] = np.transpose(data, (1, 2, 0))

                # BGR for cv2.imwrite (data is RGB from rasterio — swap channels)
                patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                fname     = f"r{row_start:05d}_c{col_start:05d}.png"
                cv2.imwrite(str(out_dir / fname), patch_bgr)
                saved += 1

    print(f"\nSaved     : {saved} patches → {out_dir.relative_to(ROOT)}")
    print(f"Skipped   : {skipped} (mask coverage < {min_cov})")
    print(f"""
Next steps:
  1. Annotate parcels in labelme:
         python steps/02_annotate/annotate_parcels.py --sheet {sheet_id}
     Draw polygon shapes with label  "parcel"  around each land parcel.
     You don't need to annotate every patch — 30-100 well-drawn parcels
     covering a variety of sizes and boundary types is enough to start.

  2. Export binary masks:
         python steps/02_annotate/export_masks.py --sheet {sheet_id} \\
             --patches-dir {out_dir.relative_to(ROOT)} \\
             --json-dir    {out_dir.relative_to(ROOT)}

  3. Fine-tune SAM:
         python steps/03_finetune/parcels/train.py --sheet {sheet_id}

  4. Predict (weights are picked up automatically):
         python steps/04_predict/parcels/parcel_predict.py --sheet {sheet_id}
""")


if __name__ == "__main__":
    main()
