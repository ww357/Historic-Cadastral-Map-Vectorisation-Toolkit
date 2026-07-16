"""
Train the DashedLineUNet on Gaussian heatmap targets, pooled across every
annotated sheet.

Unlike the boundary U-Net (steps/03_finetune/lines/train.py), this is not a
per-sheet fine-tune of a pre-existing base model — it IS the base model
training. The annotated "dashed" dataset is small (order of a hundred
patches) and spans multiple sheets, and pooling across sheets is what was
actually validated to work; a per-sheet working/iterative two-track split
(like the boundary lines pipeline) can be added later once enough sheets
have been annotated to make sheet-specific fine-tuning worthwhile.

Two-phase training:
  Phase 1 — encoder frozen, decoder trained alone.
  Phase 2 — partial unfreeze from dashed.unfreeze_from_layer up, lower LR.
            (NOT a full unfreeze — see models/DashedLineUNet/architecture.py
            for why that destabilised training on this dataset size.)

Data expected
-------------
    data/patches/images/<SHEET_ID>/*.png                    — from step 01_patchify
    data/annotations/dashed/<SHEET_ID>/gaussian/*.npy        — from gaussian_masks.py
                                                                (run automatically if missing)

Outputs
-------
    models/base/dashed/best.weights.h5    — base weights (auto-selected by predict.py)
    models/logs/dashed_pretrain_metrics.csv
    models/logs/dashed_training_curves.png

Usage
-----
    conda activate lines
    python steps/03_finetune/dashed/train.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.DashedLineUNet.architecture import (
    build_unet, preprocess_image, make_combined_loss, line_iou, unfreeze_from,
)

import tensorflow as tf

try:
    import albumentations as A
    USE_ALBUMENTATIONS = True
except ImportError:
    USE_ALBUMENTATIONS = False
    print("albumentations not found - using basic flip augmentation instead.")
    print("  Install with:  pip install albumentations")


def load_config() -> dict:
    p = ROOT / "config.yaml"
    if not p.exists():
        sys.exit(f"config.yaml not found at {p}")
    return yaml.safe_load(p.read_text())


# ---------------------------------------------------------------------------
# Data loading — pooled across every sheet with dashed annotations
# ---------------------------------------------------------------------------

def collect_pairs(cfg: dict) -> list[tuple[Path, Path]]:
    ann_root = ROOT / cfg["paths"]["annotations"] / "dashed"
    patches_root = ROOT / cfg["paths"]["patches"] / "images"

    if not ann_root.exists():
        return []

    pairs = []
    for sheet_dir in sorted(ann_root.iterdir()):
        if not sheet_dir.is_dir():
            continue
        gaussian_dir = sheet_dir / "gaussian"
        if not gaussian_dir.exists():
            continue
        for npy_path in sorted(gaussian_dir.glob("*.npy")):
            patch_id = npy_path.stem
            img_path = patches_root / sheet_dir.name / f"{patch_id}.png"
            if img_path.exists():
                pairs.append((img_path, npy_path))
    return pairs


def load_pair(img_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    mask = np.load(mask_path).astype(np.float32)
    return img, mask


if USE_ALBUMENTATIONS:
    _aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=15, p=0.4),
        A.ElasticTransform(alpha=30, sigma=4, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.GaussNoise(var_limit=(5, 25), p=0.3),
        A.Blur(blur_limit=3, p=0.2),
    ], additional_targets={"mask": "mask"})

    def augment(img, mask):
        result = _aug(image=img, mask=mask)
        return result["image"], result["mask"]
else:
    def augment(img, mask):
        if random.random() > 0.5:
            img, mask = np.fliplr(img).copy(), np.fliplr(mask).copy()
        if random.random() > 0.5:
            img, mask = np.flipud(img).copy(), np.flipud(mask).copy()
        return img, mask


class DashedLineDataset:
    def __init__(self, pairs, img_size, training=True):
        self.pairs = pairs
        self.img_size = img_size
        self.training = training

    def _generator(self):
        indices = list(range(len(self.pairs)))
        if self.training:
            random.shuffle(indices)
        for i in indices:
            img_path, mask_path = self.pairs[i]
            img, mask = load_pair(img_path, mask_path)
            if self.training:
                img, mask = augment(img, mask)
            img = preprocess_image(img)
            mask = mask[..., np.newaxis]
            yield img, mask

    def as_dataset(self, batch_size):
        ds = tf.data.Dataset.from_generator(
            self._generator,
            output_signature=(
                tf.TensorSpec(shape=(self.img_size, self.img_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(self.img_size, self.img_size, 1), dtype=tf.float32),
            ),
        )
        if self.training:
            ds = ds.shuffle(buffer_size=min(50, len(self.pairs)))
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()
    dcfg = cfg["dashed"]

    seed = int(dcfg["seed"])
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("GPU:", gpus[0].name if gpus else "CPU only")

    pairs = collect_pairs(cfg)
    if not pairs:
        print("No dashed-annotated patches found. Building them from labelme JSON...\n")
        import subprocess
        result = subprocess.run(
            [sys.executable,
             str(ROOT / "steps" / "03_finetune" / "dashed" / "gaussian_masks.py")],
            check=False,
        )
        if result.returncode != 0:
            sys.exit("gaussian_masks.py failed — check your annotations and try again.")
        pairs = collect_pairs(cfg)
    if not pairs:
        sys.exit(
            'No "dashed" annotations found anywhere under '
            f"{ROOT / cfg['paths']['annotations'] / 'dashed'}.\n"
            'Annotate at least one sheet with the "dashed" label first '
            "(steps/02_annotate/annotate.py)."
        )

    sheets = sorted({p.parent.parent.name for p, _ in pairs})
    print(f"Found {len(pairs)} patch(es) across {len(sheets)} sheet(s): {', '.join(sheets)}")

    train_pairs, val_pairs = train_test_split(
        pairs, test_size=dcfg["val_split"], random_state=seed
    )
    print(f"Train: {len(train_pairs)}  |  Val: {len(val_pairs)}")

    img_size = int(dcfg["img_size"])
    batch_size = int(dcfg["batch_size"])
    train_ds = DashedLineDataset(train_pairs, img_size, training=True).as_dataset(batch_size)
    val_ds   = DashedLineDataset(val_pairs,   img_size, training=False).as_dataset(batch_size)

    model, backbone = build_unet(img_size=img_size)
    loss_fn = make_combined_loss(float(dcfg["foreground_weight"]))

    out_dir = ROOT / cfg["paths"]["models_base"] / "dashed"
    logs_dir = ROOT / cfg["paths"]["logs"]
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = out_dir / "best.weights.h5"

    lr = float(dcfg["learning_rate"])
    epochs = int(dcfg["epochs"])
    freeze_epochs = min(15, epochs // 3)
    finetune_epochs = epochs - freeze_epochs

    # ---- Phase 1: frozen encoder ------------------------------------------
    backbone.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss_fn,
                  metrics=[line_iou(0.5)])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_weights_path), monitor="val_loss",
            save_best_only=True, save_weights_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(logs_dir / "dashed_pretrain_metrics.csv")),
    ]

    print(f"\n-- Phase 1: frozen encoder ({freeze_epochs} epochs) --")
    history1 = model.fit(train_ds, validation_data=val_ds, epochs=freeze_epochs,
                          callbacks=callbacks)

    # ---- Phase 2: partial unfreeze -----------------------------------------
    print(f"\n-- Phase 2: partial fine-tune ({finetune_epochs} epochs) --")
    bs2 = int(dcfg["batch_size_finetune"])
    train_ds2 = DashedLineDataset(train_pairs, img_size, training=True).as_dataset(bs2)
    val_ds2   = DashedLineDataset(val_pairs,   img_size, training=False).as_dataset(bs2)

    unfreeze_from(backbone, dcfg["unfreeze_from_layer"])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr / float(dcfg["finetune_lr_divisor"])),
                  loss=loss_fn, metrics=[line_iou(0.5)])

    history2 = model.fit(train_ds2, validation_data=val_ds2, epochs=finetune_epochs,
                          callbacks=callbacks)

    # ---- Plot combined curve ------------------------------------------------
    import matplotlib.pyplot as plt
    combined = {k: history1.history[k] + history2.history[k] for k in history1.history}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(combined["loss"], label="train loss")
    axes[0].plot(combined["val_loss"], label="val loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")
    iou_key = [k for k in combined if k.startswith("iou") and not k.startswith("val")][0]
    axes[1].plot(combined[iou_key], label="train IoU")
    axes[1].plot(combined["val_" + iou_key], label="val IoU")
    axes[1].set_title("IoU @ 0.5"); axes[1].legend(); axes[1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(logs_dir / "dashed_training_curves.png", dpi=150)
    plt.close()

    if best_weights_path.exists():
        model.load_weights(str(best_weights_path))
    print(f"\nDone. Best weights -> {best_weights_path.relative_to(ROOT)}")
    print(f"Training curve      -> {(logs_dir / 'dashed_training_curves.png').relative_to(ROOT)}")
    print(
        "\nNext step: run prediction then mend in QGIS:\n"
        "  python steps/04_predict/dashed/predict.py --sheet SHEET_ID\n"
        "  python steps/05_vectorise/dashed/vectorise.py --sheet SHEET_ID"
    )


if __name__ == "__main__":
    main()
