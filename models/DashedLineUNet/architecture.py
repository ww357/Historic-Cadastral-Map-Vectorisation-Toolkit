"""
U-Net with EfficientNetB0 encoder for dashed/gappy boundary-line detection.

Unlike ImprovedLinearUNet (binary Path-F1 segmentation), this model regresses
a continuous Gaussian distance-transform heatmap (1.0 on the line centre,
falling off smoothly with distance) rather than a binary mask. The blur lets
the model reproduce a human annotator's continuous linestrip through small
ink gaps, which is what actually bridges dashes into a traceable line.

Two things here matter more than they look and were the cause of a full
training collapse before they were fixed:

  1. EfficientNetB0 has its own Rescaling/Normalization layers built in and
     expects raw [0,255] pixel input. Do not divide by 255 before feeding it.
  2. UpSampling2D must use interpolation="nearest", not "bilinear". Bilinear
     resize lowers to ResizeBilinear/ResizeBilinearGrad, which forces
     float32 gradients that clash with mixed_float16 under XLA.

Known limitation (see steps/03_finetune/dashed/train.py docstring): the model
detects on-line pixels well but does not yet discriminate "the annotated
dashed line" from other faint boundary lines sharing a patch — precision
degrades in visually cluttered patches. Threshold accordingly (see
config.yaml dashed.predict_threshold) and expect more mending than the
boundary lines layer until hard-negative training data is added.
"""

import tensorflow as tf


def conv_block(x, filters, name_prefix):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same",
                                kernel_initializer="he_normal",
                                name=f"{name_prefix}_conv1")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = tf.keras.layers.Activation("relu", name=f"{name_prefix}_relu1")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same",
                                kernel_initializer="he_normal",
                                name=f"{name_prefix}_conv2")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = tf.keras.layers.Activation("relu", name=f"{name_prefix}_relu2")(x)
    return x


def upsample_block(x, skip, filters, name_prefix):
    # "nearest", not "bilinear" — see module docstring.
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest",
                                      name=f"{name_prefix}_up")(x)
    if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
        skip = tf.keras.layers.Resizing(x.shape[1], x.shape[2],
                                         name=f"{name_prefix}_resize")(skip)
    x = tf.keras.layers.Concatenate(name=f"{name_prefix}_cat")([x, skip])
    x = conv_block(x, filters, name_prefix)
    return x


def build_unet(img_size: int = 512):
    """
    EfficientNetB0 encoder -> U-Net decoder -> 1-channel sigmoid heatmap.

    Returns (model, backbone). `backbone` is exposed so callers can freeze/
    unfreeze encoder stages during a two-phase fine-tune (see train.py).
    """
    input_shape = (img_size, img_size, 3)
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=inputs,
    )

    s1     = backbone.get_layer("block2a_expand_activation").output
    s2     = backbone.get_layer("block3a_expand_activation").output
    s3     = backbone.get_layer("block4a_expand_activation").output
    s4     = backbone.get_layer("block6a_expand_activation").output
    bridge = backbone.get_layer("top_activation").output

    d1 = upsample_block(bridge, s4, 256, "dec1")
    d2 = upsample_block(d1,    s3, 128, "dec2")
    d3 = upsample_block(d2,    s2,  64, "dec3")
    d4 = upsample_block(d3,    s1,  32, "dec4")

    d5 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest",
                                       name="dec5_up")(d4)
    d5 = conv_block(d5, 16, "dec5")

    # Safety resize for odd input sizes; no-op (and so skipped) at img_size=512.
    if d5.shape[1] != input_shape[0] or d5.shape[2] != input_shape[1]:
        d5 = tf.keras.layers.Resizing(input_shape[0], input_shape[1],
                                       name="final_resize")(d5)

    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid",
                                      dtype="float32", name="output")(d5)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="DashedLineUNet")
    return model, backbone


def preprocess_image(img_uint8):
    """Cast a uint8 HxWx3 image to float32 for the network.

    EfficientNetB0 has its own Rescaling/Normalization layers and expects
    raw [0,255] input — do NOT divide by 255 here (see module docstring).
    """
    return img_uint8.astype("float32")


def make_combined_loss(foreground_weight: float):
    """MSE (up-weighted on-line pixels) + 0.5 * (1 - SSIM).

    foreground_weight up-weights the squared error on on-line pixels: the
    Gaussian heatmap target is >99% background, so plain MSE lets "predict
    near-zero everywhere" sit at a strong local minimum. This is what
    actually caused the original training collapse, not architecture size.
    """
    def weighted_mse(y_true, y_pred):
        weight = 1.0 + foreground_weight * y_true
        return tf.reduce_mean(weight * tf.square(y_true - y_pred))

    def ssim_loss(y_true, y_pred):
        return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

    def combined_loss(y_true, y_pred):
        return weighted_mse(y_true, y_pred) + 0.5 * ssim_loss(y_true, y_pred)

    return combined_loss


def line_iou(threshold=0.5):
    """IoU of thresholded predictions vs targets (at a given heatmap cutoff)."""
    def _iou(y_true, y_pred):
        y_t = tf.cast(y_true > threshold, tf.float32)
        y_p = tf.cast(y_pred > threshold, tf.float32)
        intersection = tf.reduce_sum(y_t * y_p)
        union        = tf.reduce_sum(y_t) + tf.reduce_sum(y_p) - intersection
        return (intersection + 1e-6) / (union + 1e-6)
    _iou.__name__ = f"iou_{threshold}"
    return _iou


def unfreeze_from(backbone, layer_name: str):
    """Unfreeze backbone layers from `layer_name` onward; freeze everything before it.

    Used for the two-phase fine-tune: unfreezing the whole EfficientNetB0
    destabilised training on ~150 patches (loss spiked at the unfreeze epoch
    and never recovered). Keeping the early, generic low-level blocks frozen
    and only fine-tuning the later blocks + decoder fixed it.
    """
    backbone.trainable = True
    freeze = True
    for layer in backbone.layers:
        if layer.name == layer_name:
            freeze = False
        layer.trainable = not freeze
