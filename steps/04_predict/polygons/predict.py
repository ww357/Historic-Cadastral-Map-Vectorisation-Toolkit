"""
Run MapSAM inference on all patches for a given sheet and feature class.

Any feature label used during annotation can be predicted — there is no fixed
feature list.  Pass --feature with the same label name used in labelme.

Reads  : data/patches/images/<SHEET_ID>/*.png
Writes : data/predictions/<FEATURE>/<SHEET_ID>/*.png  — 512px binary masks (0/255)

MapSAM operates at full 512px resolution.  The model decoder produces 128×128
low-resolution logits which are upsampled to 512×512 for the output masks.

Weight search order (same as train.py):
    1. --weights CLI argument (explicit override)
    2. models/finetuned/mapsam_<feature>*_best.pth  (most recent fine-tuned)
    3. models/base/MapSAM/<feature>/                (feature-specific base weights)
    4. models/base/MapSAM/origional_weights/        (generic SAM DoRA fallback)

Usage:
    python predict.py --sheet Timberscombe --feature water
    python predict.py --sheet Timberscombe --feature building --weights models/finetuned/mapsam_building_v1_best.pth
    python predict.py --sheet Timberscombe --feature water --threshold 0.4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

ROOT       = Path(__file__).resolve().parents[3]
MAPSAM_DIR = ROOT / "models" / "MapSAM"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MAPSAM_DIR))

from sam_dora_image_encoder import DoRA_Sam      # noqa: E402
from segment_anything import sam_model_registry  # noqa: E402


def load_config() -> dict:
    p = ROOT / "config.yaml"
    if not p.exists():
        sys.exit(f"config.yaml not found at {p}")
    return yaml.safe_load(p.read_text())


def resolve_weights(args_weights, feature: str, finetuned_dir: Path,
                    mapsam_base_dir: Path) -> Path:
    """
    Search order:
      1. --weights CLI argument (explicit override)
      2. Most recent mapsam_<feature>*_best.pth in models/finetuned/
      3. Most recent *.pth in models/base/MapSAM/<feature>/
      4. Most recent *.pth in models/base/MapSAM/origional_weights/
    """
    if args_weights:
        p = Path(args_weights)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            return p
        sys.exit(f"Weights not found: {p}")

    candidates = sorted(
        finetuned_dir.rglob(f"mapsam_{feature}*_best.pth"),
        key=lambda p: p.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]

    feature_base = mapsam_base_dir / feature
    if feature_base.exists():
        candidates = sorted(feature_base.rglob("*.pth"), key=lambda p: p.stat().st_mtime)
        if candidates:
            return candidates[-1]

    fallback_dir = mapsam_base_dir / "origional_weights"
    if fallback_dir.exists():
        candidates = sorted(fallback_dir.rglob("*.pth"), key=lambda p: p.stat().st_mtime)
        if candidates:
            return candidates[-1]

    sys.exit(
        f"No DoRA weights found for feature '{feature}'.\n"
        f"  Searched (finetuned) : {finetuned_dir}\n"
        f"  Searched (feature)   : {feature_base}\n"
        f"  Searched (fallback)  : {fallback_dir}\n"
        "Pass --weights <path> to specify explicitly."
    )


def load_patch_rgb(path: Path) -> np.ndarray:
    """Load a patch as a (3, H, W) float32 array in [0, 1], converting grayscale if needed."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0  # (3, H, W)


@torch.no_grad()
def run_batch(net, batch_np: list[np.ndarray], img_size: int,
              multimask_output: bool, threshold: float) -> list[np.ndarray]:
    """
    Forward a list of (3, H, W) float32 arrays through the model.
    Returns a list of (H, W) uint8 binary masks (0 or 255).
    """
    imgs = torch.from_numpy(np.stack(batch_np)).cuda()  # (B, 3, H, W)
    outputs, _ = net(imgs, multimask_output, img_size)

    # low_res_logits: (B, 1, 128, 128) — upsample to full patch resolution
    logits = outputs['low_res_logits']  # (B, num_masks, 128, 128)
    if logits.shape[1] > 1:
        # Multi-mask mode: take the mask with the highest max logit
        logits = logits[:, logits.amax(dim=(-2, -1)).argmax(dim=1), :, :]
        logits = logits.unsqueeze(1)

    upsampled = F.interpolate(logits, size=(img_size, img_size),
                              mode='bilinear', align_corners=False)  # (B, 1, H, W)
    preds = (torch.sigmoid(upsampled.squeeze(1)) > threshold).cpu().numpy()  # (B, H, W)

    return [(p.astype(np.uint8) * 255) for p in preds]


def predict(sheet_id: str, feature: str, weights_arg: str | None,
            threshold_override: float | None, batch_size_override: int | None):
    cfg   = load_config()
    mcfg  = cfg["mapsam"]
    paths = cfg["paths"]

    img_size         = int(mcfg["img_size"])
    multimask_output = int(mcfg["num_classes"]) > 1
    threshold        = threshold_override or float(mcfg.get("predict_threshold", 0.5))
    batch_size       = batch_size_override or int(mcfg.get("predict_batch_size", 4))

    patches_dir     = ROOT / paths["patches"] / "images" / sheet_id
    out_dir         = ROOT / paths["predictions"] / feature / sheet_id
    finetuned_dir   = ROOT / paths["models_finetuned"]
    mapsam_base_dir = ROOT / paths["models_base"] / "MapSAM"

    if not patches_dir.exists():
        sys.exit(f"Patches not found: {patches_dir}  — run 01_patchify first.")

    # Patches that have manual annotations are skipped — the stitch step will
    # place the annotation mask directly, which is already ground-truth quality.
    ann_mask_dir = ROOT / paths["annotations"] / feature / sheet_id / "masks"
    annotated    = {p.stem for p in ann_mask_dir.glob("*.png")} if ann_mask_dir.exists() else set()

    weights_path = resolve_weights(weights_arg, feature, finetuned_dir, mapsam_base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sheet     : {sheet_id}")
    print(f"Feature   : {feature}")
    print(f"img_size  : {img_size}px (no sub-tile splitting)")
    print(f"Threshold : {threshold}")
    print(f"Weights   : {weights_path.relative_to(ROOT)}")
    if annotated:
        print(f"Skipping  : {len(annotated)} annotated patches (annotation mask used in stitch)")
    print()

    # ---- Load model -----------------------------------------------------------
    sam_ckpt = mapsam_base_dir / "origional_weights" / "sam_vit_b_01ec64.pth"
    if not sam_ckpt.exists():
        sys.exit(f"SAM base checkpoint not found: {sam_ckpt}")

    sam, _ = sam_model_registry[mcfg["vit_name"]](
        image_size  = img_size,
        num_classes = mcfg["num_classes"],
        checkpoint  = str(sam_ckpt),
        pixel_mean  = [0, 0, 0],
        pixel_std   = [1, 1, 1],
    )
    net = DoRA_Sam(sam, mcfg["rank"]).cuda()
    net.load_dora_parameters(str(weights_path))
    net.eval()
    print("Model loaded.\n")

    # ---- Inference ------------------------------------------------------------
    patch_paths = sorted(patches_dir.glob("*.png"))
    to_predict  = [p for p in patch_paths if p.stem not in annotated]
    print(f"{len(patch_paths)} total patches  →  {len(to_predict)} to predict")

    failed = 0
    buffer: list[tuple[Path, np.ndarray]] = []

    def flush_buffer():
        nonlocal failed
        if not buffer:
            return
        paths_b, arrays_b = zip(*buffer)
        try:
            masks = run_batch(net, list(arrays_b), img_size, multimask_output, threshold)
            for p, m in zip(paths_b, masks):
                from PIL import Image
                Image.fromarray(m, mode="L").save(out_dir / p.name)
        except Exception as e:
            print(f"\nWarning: batch failed — {e}")
            failed += len(buffer)
        buffer.clear()

    for patch_path in tqdm(to_predict, unit="patch"):
        try:
            arr = load_patch_rgb(patch_path)
        except Exception as e:
            print(f"\nWarning: could not load {patch_path.name}: {e}")
            failed += 1
            continue

        buffer.append((patch_path, arr))
        if len(buffer) >= batch_size:
            flush_buffer()

    flush_buffer()

    saved = len(to_predict) - failed
    print(f"\nDone  ({saved} predicted, {len(annotated)} annotation-only, {failed} failed)")
    print(f"  -> {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MapSAM inference on patchified map sheet."
    )
    parser.add_argument("--sheet",      required=True,
                        help="Sheet ID (subdirectory under patches/images/)")
    parser.add_argument("--feature",    required=True,
                        help="Feature class — any label used in labelme annotations "
                             "(e.g. water, building, vegetation)")
    parser.add_argument("--weights",    default=None,
                        help="DoRA .pth weights file (default: auto-selects by search order)")
    parser.add_argument("--threshold",  type=float, default=None,
                        help="Sigmoid threshold for binary mask (default: from config or 0.5)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Patches per forward pass (default: from config or 4)")
    args = parser.parse_args()
    predict(args.sheet, args.feature, args.weights, args.threshold, args.batch_size)
