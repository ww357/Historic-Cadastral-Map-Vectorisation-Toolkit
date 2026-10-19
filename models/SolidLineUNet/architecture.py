"""
Attention U-Net with ASPP bottleneck for thin linear feature segmentation.

Loss options (set in config.yaml under unet.loss_type):
  combined_focal_cldice  — recommended; focal loss handles class imbalance
                           (~3-5% boundary pixels), clDice rewards connected lines
  combined_cldice        — Dice + clDice topology term, no class weighting
  dice                   — plain Dice baseline / backward-compatible
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, BatchNormalization, Activation, Dropout,
    Conv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D,
    UpSampling2D, Concatenate, Lambda, add, multiply,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from tqdm import tqdm

np.random.seed(42)
tf.random.set_seed(42)

_kinit = "he_normal"

LOSS_MAP = None  # populated after loss functions are defined


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dsc(y_true, y_pred):
    smooth = 1.
    yt, yp = K.flatten(y_true), K.flatten(y_pred)
    return (2. * K.sum(yt * yp) + smooth) / (K.sum(yt) + K.sum(yp) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dsc(y_true, y_pred)

def _confusion(y_true, y_pred):
    smooth = 1
    yp_pos = K.clip(y_pred, 0, 1)
    yp_neg = 1 - yp_pos
    y_pos  = K.clip(y_true, 0, 1)
    y_neg  = 1 - y_pos
    tp = K.sum(y_pos * yp_pos)
    fp = K.sum(y_neg * yp_pos)
    fn = K.sum(y_pos * yp_neg)
    return (tp + smooth) / (tp + fp + smooth), (tp + smooth) / (tp + fn + smooth)

def tp(y_true, y_pred):
    smooth = 1
    yp = K.round(K.clip(y_pred, 0, 1))
    yt = K.round(K.clip(y_true, 0, 1))
    return (K.sum(yt * yp) + smooth) / (K.sum(yt) + smooth)

def tn(y_true, y_pred):
    smooth = 1
    yp_pos = K.round(K.clip(y_pred, 0, 1))
    yp_neg = 1 - yp_pos
    yt_neg = 1 - K.round(K.clip(y_true, 0, 1))
    return (K.sum(yt_neg * yp_neg) + smooth) / (K.sum(yt_neg) + smooth)

def prec(y_true, y_pred):
    return _confusion(y_true, y_pred)[0]

def recall(y_true, y_pred):
    return _confusion(y_true, y_pred)[1]


# ---------------------------------------------------------------------------
# clDice loss
# Topology-preserving loss for line continuity (Shit et al., CVPR 2021).
# Standard Dice treats each pixel independently — a line with a 3-pixel gap
# scores nearly the same as a fully connected one. clDice computes Dice on
# the *skeleton* of each prediction, explicitly penalising breaks in lines.
# ---------------------------------------------------------------------------

def _soft_erode(x):
    return -tf.nn.max_pool2d(-x, ksize=[1,3,3,1], strides=[1,1,1,1], padding="SAME")

def _soft_dilate(x):
    return tf.nn.max_pool2d(x, ksize=[1,3,3,1], strides=[1,1,1,1], padding="SAME")

def _soft_skel(x, iters=3):
    x1   = _soft_dilate(_soft_erode(x))
    skel = tf.nn.relu(x - x1)
    for _ in range(iters - 1):
        x     = _soft_erode(x)
        x1    = _soft_dilate(_soft_erode(x))
        delta = tf.nn.relu(x - x1)
        skel  = skel + tf.nn.relu(delta - skel * delta)
    return skel

def clDice(y_true, y_pred, smooth=1.):
    sp = _soft_skel(y_pred)
    st = _soft_skel(K.cast(y_true, "float32"))
    t_prec = (K.sum(sp * K.cast(y_true, "float32")) + smooth) / (K.sum(sp) + smooth)
    t_sens = (K.sum(st * y_pred) + smooth) / (K.sum(st) + smooth)
    return 2.0 * t_prec * t_sens / (t_prec + t_sens + K.epsilon())

def combined_clDice_loss(y_true, y_pred):
    return 0.5 * dice_loss(y_true, y_pred) + 0.5 * (1.0 - clDice(y_true, y_pred))


# ---------------------------------------------------------------------------
# Focal loss — for class-imbalanced segmentation (~3-5% boundary pixels).
# Down-weights easy background pixels via (1-p_t)^gamma so training focuses
# on hard/uncertain boundary pixels. alpha=0.75 gives boundary pixels 3x
# the gradient weight of background (compensates for ~1:27 class imbalance).
# ---------------------------------------------------------------------------

def focal_loss(y_true, y_pred, alpha=0.75, gamma=2.0):
    y_pred  = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    bce     = -(y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))
    p_t     = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
    alpha_t = y_true * alpha  + (1.0 - y_true) * (1.0 - alpha)
    return K.mean(alpha_t * K.pow(1.0 - p_t, gamma) * bce)

def combined_focal_clDice_loss(y_true, y_pred):
    return 0.5 * focal_loss(y_true, y_pred, alpha=0.75, gamma=2.0) + \
           0.5 * (1.0 - clDice(y_true, y_pred))


LOSS_MAP = {
    "combined_focal_cldice": combined_focal_clDice_loss,
    "combined_cldice":       combined_clDice_loss,
    "dice":                  dice_loss,
}


# ---------------------------------------------------------------------------
# Model building blocks
# ---------------------------------------------------------------------------

def _conv_block(x, filters, is_bn, name):
    for i in (1, 2):
        x = Conv2D(filters, (3,3), strides=(1,1), kernel_initializer=_kinit,
                   padding="same", name=f"{name}_{i}")(x)
        if is_bn:
            x = BatchNormalization(name=f"{name}_{i}_bn")(x)
        x = Activation("relu", name=f"{name}_{i}_act")(x)
    return x

def _gating_signal(x, is_bn, name):
    s = K.int_shape(x)[3]
    x = Conv2D(s, (1,1), strides=(1,1), padding="same", name=f"{name}_conv")(x)
    if is_bn:
        x = BatchNormalization(name=f"{name}_bn")(x)
    return Activation("relu", name=f"{name}_act")(x)

def _attn_gate(x, g, inter_shape, name):
    sx, sg = K.int_shape(x), K.int_shape(g)
    theta  = Conv2D(inter_shape, (2,2), strides=(2,2), padding="same", name=f"xl{name}")(x)
    st     = K.int_shape(theta)
    phi_g  = Conv2DTranspose(inter_shape, (3,3),
                             strides=(st[1] // sg[1], st[2] // sg[2]),
                             padding="same", name=f"g_up{name}")(
                 Conv2D(inter_shape, (1,1), padding="same")(g))
    psi    = Activation("sigmoid")(
                 Conv2D(1, (1,1), padding="same", name=f"psi{name}")(
                     Activation("relu")(add([phi_g, theta]))))
    sp     = K.int_shape(psi)
    up_psi = Lambda(lambda x: K.repeat_elements(x, sx[3], axis=3), name=f"psi_up{name}")(
                 UpSampling2D(size=(sx[1] // sp[1], sx[2] // sp[2]))(psi))
    y = multiply([up_psi, x], name=f"q_attn{name}")
    return BatchNormalization(name=f"q_attn_bn{name}")(
               Conv2D(sx[3], (1,1), padding="same", name=f"q_attn_conv{name}")(y))

def _aspp(x, out_ch, name):
    branches = []
    for i, rate in enumerate([1, 6, 12, 18], 1):
        b = Conv2D(out_ch, (1,1) if rate == 1 else (3,3),
                   kernel_initializer=_kinit, padding="same",
                   dilation_rate=rate, name=f"{name}_conv{i}")(x)
        b = Activation("relu")(BatchNormalization(name=f"{name}_bn{i}")(b))
        branches.append(b)
    # global context branch
    g = AveragePooling2D(pool_size=(1,1))(x)
    g = Activation("relu")(BatchNormalization(name=f"{name}_bn5")(
            Conv2D(out_ch, (1,1), kernel_initializer=_kinit,
                   padding="same", name=f"{name}_conv5")(g)))
    ih, iw = K.int_shape(x)[1], K.int_shape(x)[2]
    gh, gw = K.int_shape(g)[1], K.int_shape(g)[2]
    g = UpSampling2D(size=(ih // gh, iw // gw))(g)
    branches.append(g)
    out = Concatenate(axis=3)(branches)
    out = Activation("relu")(BatchNormalization(name=f"{name}_bn_final")(
              Conv2D(out_ch, (1,1), kernel_initializer=_kinit,
                     padding="same", name=f"{name}_conv_final")(out)))
    return Dropout(0.5)(out)


def build_attn_unet(input_size):
    """Build the Attention U-Net with ASPP bottleneck. input_size = (H, W, C)."""
    inputs = Input(shape=input_size)

    c1 = Dropout(0.2, name="drop1")(_conv_block(inputs,  32, True, "conv1"))
    c2 = Dropout(0.2, name="drop2")(_conv_block(MaxPooling2D()(c1), 32, True, "conv2"))
    c3 = Dropout(0.2, name="drop3")(_conv_block(MaxPooling2D()(c2), 64, True, "conv3"))
    c4 = Dropout(0.2, name="drop4")(_conv_block(MaxPooling2D()(c3), 64, True, "conv4"))

    center = _aspp(MaxPooling2D()(c4), 128, "center")

    g1  = _gating_signal(center, True, "g1")
    up1 = Concatenate(name="up1")([
              Conv2DTranspose(32, (3,3), strides=(2,2), padding="same",
                              activation="relu", kernel_initializer=_kinit)(center),
              _attn_gate(c4, g1, 128, "_1")])

    g2  = _gating_signal(up1, True, "g2")
    up2 = Concatenate(name="up2")([
              Conv2DTranspose(64, (3,3), strides=(2,2), padding="same",
                              activation="relu", kernel_initializer=_kinit)(up1),
              _attn_gate(c3, g2, 64, "_2")])

    g3  = _gating_signal(up2, True, "g3")
    up3 = Concatenate(name="up3")([
              Conv2DTranspose(32, (3,3), strides=(2,2), padding="same",
                              activation="relu", kernel_initializer=_kinit)(up2),
              _attn_gate(c2, g3, 32, "_3")])

    up4 = Concatenate(name="up4")([
              Conv2DTranspose(32, (3,3), strides=(2,2), padding="same",
                              activation="relu", kernel_initializer=_kinit)(up3),
              c1])

    out = Conv2D(1, (1,1), activation="sigmoid",
                 kernel_initializer=_kinit, name="final")(up4)
    return Model(inputs=[inputs], outputs=[out])


def build_model(inference_size, channels, loss_type):
    """Compile and return the model. loss_type must be a key in LOSS_MAP."""
    loss_fn = LOSS_MAP.get(loss_type)
    if loss_fn is None:
        raise ValueError(f"Unknown loss '{loss_type}'. Choose from: {list(LOSS_MAP)}")
    model = build_attn_unet((inference_size, inference_size, channels))
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=loss_fn,
                  metrics=[dsc, clDice, tp, tn, prec, recall])
    return model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _resolve_dirs(data_path, split, need_masks):
    """Resolve image/mask folders for both supported dataset layouts:
      split layout:  <root>/{train,test}/  +  <root>/annotation/{train,test}/
      legacy layout: <root>/images/        +  <root>/masks/
    """
    import os
    split_imgs  = os.path.join(data_path, split)
    split_masks = os.path.join(data_path, "annotation", split)

    if os.path.isdir(split_imgs):
        if need_masks and not os.path.isdir(split_masks):
            raise FileNotFoundError(f"Masks dir not found: {split_masks}")
        return split_imgs, split_masks if need_masks else None

    legacy_imgs  = os.path.join(data_path, "images")
    legacy_masks = os.path.join(data_path, "masks")
    if os.path.isdir(legacy_imgs):
        if need_masks and not os.path.isdir(legacy_masks):
            raise FileNotFoundError(f"Masks dir not found: {legacy_masks}")
        return legacy_imgs, legacy_masks if need_masks else None

    raise FileNotFoundError(
        f"Cannot resolve dataset dirs in '{data_path}'. "
        "Expected: <root>/train + <root>/annotation/train  OR  <root>/images + <root>/masks"
    )


def get_data(data_path, inference_size, channels, split="train", with_masks=True):
    """Load and preprocess all images (and masks) for a dataset split.
    Returns (X, y) if with_masks=True, else X.
    """
    import os
    imgs_dir, masks_dir = _resolve_dirs(data_path, split, with_masks)
    ids = sorted(f for f in os.listdir(imgs_dir) if os.path.isfile(os.path.join(imgs_dir, f)))

    X = np.zeros((len(ids), inference_size, inference_size, channels), dtype=np.float32)
    y = np.zeros((len(ids), inference_size, inference_size, 1),        dtype=np.float32) if with_masks else None

    print(f"Loading {split} data ({len(ids)} images)...")
    for n, fname in enumerate(tqdm(ids)):
        img = load_img(os.path.join(imgs_dir, fname), color_mode="grayscale")
        arr = tf.image.resize(img_to_array(img), (inference_size, inference_size)).numpy()
        X[n] = arr / 255.0

        if with_masks:
            stem = fname.rsplit(".", 1)[0]
            mask_path = os.path.join(masks_dir, f"{stem}.png")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            mask = load_img(mask_path, color_mode="grayscale")
            y[n] = tf.image.resize(img_to_array(mask),
                                    (inference_size, inference_size)).numpy() / 255.0

    return (X, y) if with_masks else X


# ---------------------------------------------------------------------------
# Evaluation / visualisation (used in notebooks)
# ---------------------------------------------------------------------------

def plot_training_history(history_path):
    import pickle
    import matplotlib.pyplot as plt

    with open(history_path, "rb") as f:
        h = pickle.load(f)

    has_cldice = "clDice" in h
    n = 3 if has_cldice else 2
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    axes[0].plot(h["loss"], label="Train"); axes[0].plot(h["val_loss"], label="Val")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss"); axes[0].legend()

    axes[1].plot(h["dsc"], label="Train"); axes[1].plot(h["val_dsc"], label="Val")
    axes[1].set(title="Dice", xlabel="Epoch", ylabel="Dice"); axes[1].legend()

    if has_cldice:
        axes[2].plot(h["clDice"], label="Train"); axes[2].plot(h["val_clDice"], label="Val")
        axes[2].set(title="clDice", xlabel="Epoch"); axes[2].legend()

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()


def evaluate(weights_path, data_path, inference_size, channels,
             loss_type="combined_focal_cldice", split="test"):
    model = build_model(inference_size, channels, loss_type)
    model.load_weights(weights_path)
    X, y = get_data(data_path, inference_size, channels, split=split)
    results = model.evaluate(X, y, batch_size=1, verbose=1)
    names = ["loss", "dice", "clDice", "tp", "tn", "precision", "recall"]
    print("\nTest results:")
    for name, val in zip(names, results):
        print(f"  {name:<12} {val:.4f}")
    del X, y
    tf.keras.backend.clear_session()


def visualise_patch(weights_path, image_path, mask_path, inference_size, channels,
                    loss_type="combined_focal_cldice", threshold=0.95):
    import matplotlib.pyplot as plt

    model = build_model(inference_size, channels, loss_type)
    model.load_weights(weights_path)

    img  = load_img(image_path, color_mode="grayscale")
    orig = img_to_array(img)
    arr  = tf.image.resize(orig, (inference_size, inference_size)).numpy() / 255.0

    pred   = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
    binary = (pred > threshold).astype(np.uint8)

    mask_orig = img_to_array(load_img(mask_path, color_mode="grayscale"))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(orig / 255., cmap="gray");       axes[0].set_title("Input")
    axes[1].imshow(pred.squeeze(), cmap="viridis"); axes[1].set_title("Probability")
    axes[2].imshow(binary.squeeze(), cmap="gray");  axes[2].set_title(f"Binary (>{threshold})")
    axes[3].imshow(mask_orig.squeeze(), cmap="gray"); axes[3].set_title("Ground truth")
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.show()
    return pred, binary
