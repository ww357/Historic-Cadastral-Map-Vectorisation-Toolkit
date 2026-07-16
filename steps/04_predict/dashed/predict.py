"""
Run DashedLineUNet inference on all patches for a given sheet.

Reads  : data/patches/images/<SHEET_ID>/*.png            (512px patches)
Writes : data/predictions/dashed/<SHEET_ID>/*.png        (512px binary masks)

Unlike the boundary U-Net, no 2x2 tiling is needed here — the model's native
input size (dashed.img_size, 512) already matches the patch size, so each
patch is a single forward pass.

Weight search order:
  1. --weights CLI argument (explicit path)
  2. Most recently modified *.weights.h5 in models/base/dashed/

(No working/iterative two-track split yet, unlike the boundary lines
pipeline — the "dashed" model is currently a single pooled cross-sheet model
trained from steps/03_finetune/dashed/train.py. Add a working/ tier here if
per-sheet fine-tuning is introduced later.)

Patches with an existing "dashed" annotation mask are skipped — the stitch
step in vectorise.py uses that ground-truth mask directly instead, same
convention as the boundary lines pipeline.

Usage:
    conda activate lines
    python steps/04_predict/dashed/predict.py --sheet SHEET_ID
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.DashedLineUNet.architecture import build_unet, preprocess_image


def load_config() -> dict:
    p = ROOT / "config.yaml"
    if not p.exists():
        sys.exit(f"config.yaml not found at {p}")
    return yaml.safe_load(p.read_text())


def resolve_weights(weights_arg: str | None, repo_root: Path, paths_cfg: dict) -> Path:
    if weights_arg:
        p = Path(weights_arg)
        if not p.is_absolute():
            p = repo_root / p
        if p.exists():
            return p
        sys.exit(f"Weights not found: {p}")

    base_dir = repo_root / paths_cfg["models_base"] / "dashed"
    candidates = sorted(base_dir.glob("*.weights.h5"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]

    sys.exit(
        f"No weights found in {base_dir} (*.weights.h5).\n"
        "Pass --weights <path> to specify a file explicitly, or run "
        "steps/03_finetune/dashed/train.py first."
    )


def predict(sheet_id: str, weights_arg: str | None = None):
    cfg  = load_config()
    dcfg = cfg["dashed"]

    img_size  = int(dcfg["img_size"])
    threshold = float(dcfg["predict_threshold"])

    patches_dir  = ROOT / cfg["paths"]["patches"] / "images" / sheet_id
    out_dir      = ROOT / cfg["paths"]["predictions"] / "dashed" / sheet_id
    weights_path = resolve_weights(weights_arg, ROOT, cfg["paths"])

    if not patches_dir.exists():
        sys.exit(f"Patches not found: {patches_dir} — run 01_patchify first.")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Patches with a reviewed "dashed" annotation are skipped — vectorise.py's
    # stitch step uses that ground-truth mask directly (see gaussian_masks.py:
    # this includes confirmed-negative patches, not just positive ones).
    ann_mask_dir = ROOT / cfg["paths"]["annotations"] / "dashed" / sheet_id / "masks"
    annotated = {p.stem for p in ann_mask_dir.glob("*.png")} if ann_mask_dir.exists() else set()

    print(f"Sheet    : {sheet_id}")
    print(f"Threshold: {threshold}")
    print(f"Weights  : {weights_path.relative_to(ROOT)}")
    if annotated:
        print(f"Skipping : {len(annotated)} annotated patches (annotation mask used in stitch)")
    print()

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    model, _ = build_unet(img_size=img_size)
    model.load_weights(str(weights_path))
    print("Model loaded.\n")

    patch_paths = sorted(patches_dir.glob("*.png"))
    to_predict  = [p for p in patch_paths if p.stem not in annotated]
    print(f"{len(patch_paths)} total patches  ->  {len(to_predict)} to predict")

    failed = 0
    for patch_path in tqdm(to_predict, unit="patch"):
        try:
            img = np.array(Image.open(patch_path).convert("RGB"), dtype=np.uint8)
        except Exception as e:
            print(f"Warning: could not load {patch_path.name}: {e}")
            failed += 1
            continue

        if img.shape[0] != img_size or img.shape[1] != img_size:
            print(f"Warning: {patch_path.name} is {img.shape[:2]}, expected "
                  f"({img_size}, {img_size}) — skipping.")
            failed += 1
            continue

        x = preprocess_image(img)[np.newaxis]
        pred = model.predict(x, verbose=0)[0, :, :, 0]
        binary = (pred > threshold).astype(np.uint8) * 255

        Image.fromarray(binary, mode="L").save(out_dir / patch_path.name)

    saved = len(to_predict) - failed
    print(f"\nDone  ({saved} predicted, {len(annotated)} annotation-only, {failed} failed)")
    print(f"  -> {out_dir}/")
    tf.keras.backend.clear_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DashedLineUNet on a patchified map sheet.")
    parser.add_argument("--sheet", required=True, help="Sheet ID")
    parser.add_argument("--weights", default=None,
                        help="Path to weights file (default: auto-selects from models/base/dashed/)")
    args = parser.parse_args()
    predict(args.sheet, args.weights)
