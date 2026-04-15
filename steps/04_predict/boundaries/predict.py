"""
Run boundary U-Net inference on all patches for a given sheet.

Reads  : data/patches/images/<SHEET_ID>/*.png       (512px patches)
Writes : data/predictions/boundaries/<SHEET_ID>/*.png (512px binary masks)

Each 512px patch is split into a 2×2 grid of 256px sub-patches (4 total),
matching the model's training resolution exactly — no downsampling. Predictions
are reassembled into a 512px output mask aligned with the metadata CSV.

Usage:
    python predict.py --sheet SHEET_ID
"""

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

from models.unet.architecture import build_model


def load_config() -> dict:
    p = ROOT / "config.yaml"
    if not p.exists():
        sys.exit(f"config.yaml not found at {p}")
    return yaml.safe_load(p.read_text())


def split_patch(arr: np.ndarray, sub_size: int) -> tuple[list[np.ndarray], list[tuple]]:
    """Split a (H, W) array into (H/sub_size × W/sub_size) tiles.
    Returns tiles and their (row_off, col_off) positions for reassembly.
    """
    tiles, positions = [], []
    for r in range(0, arr.shape[0], sub_size):
        for c in range(0, arr.shape[1], sub_size):
            tiles.append(arr[r:r + sub_size, c:c + sub_size])
            positions.append((r, c))
    return tiles, positions


def assemble_patch(tiles: list[np.ndarray], positions: list[tuple],
                   full_size: int, sub_size: int) -> np.ndarray:
    out = np.zeros((full_size, full_size), dtype=tiles[0].dtype)
    for tile, (r, c) in zip(tiles, positions):
        out[r:r + sub_size, c:c + sub_size] = tile
    return out


def resolve_weights(weights_arg: str | None, repo_root: Path, paths_cfg: dict) -> Path:
    """
    Find weights to use. Search order:
      1. --weights CLI argument (explicit path)
      2. Most recently modified *_best.weights.h5 in models/finetuned/
      3. models/finetuned/model_weights.weights.h5
      4. Most recently modified *.weights.h5 in models/base/ (recursive)
    """
    if weights_arg:
        p = Path(weights_arg)
        if not p.is_absolute():
            p = repo_root / p
        if p.exists():
            return p
        sys.exit(f"Weights not found: {p}")

    finetuned_dir = repo_root / paths_cfg["models_finetuned"]
    base_dir      = repo_root / paths_cfg["models_base"]

    # Most recent fine-tune checkpoint
    candidates = sorted(finetuned_dir.rglob("*_best.weights.h5"),
                        key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]

    # Explicit named file in finetuned dir
    named = finetuned_dir / "model_weights.weights.h5"
    if named.exists():
        return named

    # Fall back to base weights
    candidates = sorted(base_dir.rglob("*.weights.h5"),
                        key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]

    sys.exit(
        f"No weights found. Searched:\n"
        f"  {finetuned_dir} (*_best.weights.h5)\n"
        f"  {base_dir} (*.weights.h5)\n"
        "Pass --weights <path> to specify a file explicitly."
    )


def predict(sheet_id: str, repo_root: Path, weights_arg: str | None = None):
    cfg  = load_config()
    ucfg = cfg["unet"]

    sub_size   = int(ucfg["inference_size"])
    patch_size = int(cfg["patchify"]["patch_size"])
    n_tiles    = (patch_size // sub_size) ** 2
    channels   = int(ucfg["image_channels"])
    loss_type  = ucfg["loss_type"]
    threshold  = float(ucfg["threshold"])

    patches_dir  = repo_root / cfg["paths"]["patches"] / "images" / sheet_id
    out_dir      = repo_root / cfg["paths"]["predictions"] / "boundaries" / sheet_id
    weights_path = resolve_weights(weights_arg, repo_root, cfg["paths"])

    if not patches_dir.exists():
        sys.exit(f"Patches not found: {patches_dir}  — run 01_patchify first.")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Patches that have manual annotations are skipped — the stitch step will
    # place the annotation mask directly, which is already ground-truth quality.
    boundary_label = cfg["annotation"].get("boundary_label", "boundary")
    ann_mask_dir   = repo_root / cfg["paths"]["annotations"] / boundary_label / sheet_id / "masks"
    annotated      = {p.stem for p in ann_mask_dir.glob("*.png")} if ann_mask_dir.exists() else set()

    print(f"Sheet    : {sheet_id}")
    print(f"Strategy : {patch_size}px patch → {n_tiles}×{sub_size}px tiles (no downsampling)")
    print(f"Threshold: {threshold}  |  loss: {loss_type}")
    print(f"Weights  : {weights_path.relative_to(repo_root)}")
    if annotated:
        print(f"Skipping : {len(annotated)} annotated patches (annotation mask used in stitch)")
    print()

    model = build_model(sub_size, channels, loss_type)
    model.load_weights(str(weights_path))
    print("Model loaded.\n")

    patch_paths = sorted(patches_dir.glob("*.png"))
    to_predict  = [p for p in patch_paths if p.stem not in annotated]
    print(f"{len(patch_paths)} total patches  →  {len(to_predict)} to predict")

    failed = 0
    for patch_path in tqdm(to_predict, unit="patch"):
        try:
            grey = np.array(Image.open(patch_path).convert("L"), dtype=np.float32)
        except Exception as e:
            print(f"Warning: could not load {patch_path.name}: {e}")
            failed += 1
            continue

        tiles, positions = split_patch(grey, sub_size)

        # Stack all tiles into one batch — (N, 256, 256, 1), normalised
        batch = np.stack([t[:, :, None] / 255.0 for t in tiles])

        preds = model.predict(batch, verbose=0)  # (N, 256, 256, 1)

        pred_tiles = [(pred.squeeze() > threshold).astype(np.uint8) for pred in preds]
        full_mask  = assemble_patch(pred_tiles, positions, patch_size, sub_size)

        Image.fromarray((full_mask * 255).astype(np.uint8), mode="L").save(
            out_dir / patch_path.name
        )

    saved = len(to_predict) - failed
    print(f"\nDone  ({saved} predicted, {len(annotated)} annotation-only, {failed} failed)")
    print(f"  -> {out_dir}/")
    tf.keras.backend.clear_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run boundary U-Net on patchified map sheet.")
    parser.add_argument("--sheet",   required=True, help="Sheet ID")
    parser.add_argument("--weights", default=None,
                        help="Path to weights file (default: auto-selects most recent fine-tune)")
    args = parser.parse_args()
    predict(args.sheet, ROOT, args.weights)
