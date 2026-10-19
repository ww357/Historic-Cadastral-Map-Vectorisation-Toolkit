"""
Boundary-safe augmentation for thin linear feature segmentation.

Avoids blur, posterisation, sharpness reduction, and sub-pixel resizing — all
of which erase 1-2px boundary lines. Keeps only flips, 90° rotations, and mild
brightness/contrast perturbation.

Key addition: occlusion patch augmentation pastes opaque rectangles onto the
IMAGE only (mask untouched), forcing the model to predict line continuity under
text labels, hatching, and other map features that occlude boundaries. This is
the most impactful augmentation for the line continuity problem.
"""

import tensorflow as tf
import numpy as np


class BoundaryAugmentation:
    def __init__(self, patch_size=256, probability=0.7, occlusion_p=0.6):
        self.patch_size  = patch_size
        self.probability = probability
        self.occlusion_p = occlusion_p

    def _flip(self, image, mask):
        if tf.random.uniform([]) < self.probability:
            image = tf.image.flip_left_right(image)
            mask  = tf.image.flip_left_right(mask)
        if tf.random.uniform([]) < self.probability:
            image = tf.image.flip_up_down(image)
            mask  = tf.image.flip_up_down(mask)
        return image, mask

    def _rotate90(self, image, mask):
        if tf.random.uniform([]) < self.probability:
            k     = tf.random.uniform([], 0, 4, dtype=tf.int32)
            image = tf.image.rot90(image, k=k)
            mask  = tf.image.rot90(mask,  k=k)
        return image, mask

    def _brightness_contrast(self, image, mask):
        if tf.random.uniform([]) < self.probability:
            image = tf.image.random_brightness(image, max_delta=0.08)
        if tf.random.uniform([]) < self.probability:
            image = tf.image.random_contrast(image, lower=0.88, upper=1.12)
        return tf.clip_by_value(image, 0.0, 1.0), mask

    def _occlusion(self, image, mask):
        """Paste 1–4 opaque rectangles onto image; mask unchanged.
        Light fill simulates text/labels, dark fill simulates hatching/ink.
        Elongated shape mimics text blocks — the most common occlusion source.
        """
        H, W = self.patch_size, self.patch_size
        n = tf.random.uniform([], 1, 5, dtype=tf.int32)
        for i in range(4):
            ph = tf.random.uniform([], 6,  24, dtype=tf.int32)
            pw = tf.random.uniform([], 18, 56, dtype=tf.int32)
            py = tf.random.uniform([], 0, tf.maximum(1, H - ph), dtype=tf.int32)
            px = tf.random.uniform([], 0, tf.maximum(1, W - pw), dtype=tf.int32)
            y_m = tf.logical_and(tf.range(H) >= py, tf.range(H) < py + ph)
            x_m = tf.logical_and(tf.range(W) >= px, tf.range(W) < px + pw)
            occ = tf.cast(tf.logical_and(y_m[:, None], x_m[None, :]), tf.float32)[:, :, None]
            fill = tf.cond(tf.random.uniform([]) < 0.5,
                           lambda: tf.random.uniform([], 0.78, 0.95),
                           lambda: tf.random.uniform([], 0.04, 0.28))
            patched = image * (1.0 - occ) + fill * occ
            image = tf.cond(tf.less(tf.constant(i, dtype=tf.int32), n),
                            lambda: patched, lambda: image)
        return image, mask

    def __call__(self, image, mask):
        image, mask = self._flip(image, mask)
        image, mask = self._rotate90(image, mask)
        image, mask = self._brightness_contrast(image, mask)
        if tf.random.uniform([]) < self.occlusion_p:
            image, mask = self._occlusion(image, mask)
        return image, mask


def make_augmented_dataset(X, y, batch_size, patch_size):
    aug = BoundaryAugmentation(patch_size=patch_size, probability=0.7, occlusion_p=0.6)
    ds  = tf.data.Dataset.from_tensor_slices((X, y))
    ds  = ds.shuffle(buffer_size=len(X))
    ds  = ds.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)),
                 num_parallel_calls=tf.data.AUTOTUNE)
    ds  = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
    ds  = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
