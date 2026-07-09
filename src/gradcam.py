"""Grad-CAM explainability for the DenseNet121 pneumonia model.

Grad-CAM (Gradient-weighted Class Activation Mapping) produces a heatmap
highlighting the regions of the X-ray that most influenced the model's
prediction. This makes the diagnosis interpretable and trustworthy — a
core objective of the project.
"""
from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf

from . import config


def _find_base(model: tf.keras.Model) -> tf.keras.Model:
    """Return the nested DenseNet base model inside the classifier."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise RuntimeError("Could not locate the DenseNet base model inside the classifier.")


def _head_layers(model: tf.keras.Model, base: tf.keras.Model) -> list:
    """The classification-head layers that sit on top of the base, in order."""
    return [
        layer
        for layer in model.layers
        if layer is not base and not isinstance(layer, tf.keras.layers.InputLayer)
    ]


def compute_heatmap(model: tf.keras.Model, preprocessed_image: np.ndarray,
                    layer_name: str = config.GRADCAM_LAYER) -> np.ndarray:
    """Compute a normalised Grad-CAM heatmap (values in [0, 1]).

    ``preprocessed_image`` must be a single image already run through the
    DenseNet ``preprocess_input`` step, shaped ``(H, W, 3)`` or ``(1, H, W, 3)``.

    Because the DenseNet backbone is a *nested* Keras model, we can't tap an
    internal layer via a single functional Model (Keras 3 rejects it). Instead
    we map the image to the base's final feature map, then re-apply the
    classification head on top of that tensor inside the gradient tape — so the
    gradient path from the pneumonia score back to the feature map is intact.
    """
    if preprocessed_image.ndim == 3:
        preprocessed_image = preprocessed_image[np.newaxis, ...]
    x_in = tf.convert_to_tensor(preprocessed_image, dtype=tf.float32)

    base = _find_base(model)
    # Image -> final convolutional feature maps (7x7x1024 for DenseNet121).
    feature_model = tf.keras.Model(base.inputs, base.output)
    head = _head_layers(model, base)

    with tf.GradientTape() as tape:
        conv_output = feature_model(x_in, training=False)
        tape.watch(conv_output)
        x = conv_output
        for layer in head:                 # gap -> dropout -> dense(sigmoid)
            x = layer(x, training=False)
        class_channel = x[:, 0]            # P(pneumonia)

    grads = tape.gradient(class_channel, conv_output)
    # Global-average-pool the gradients -> importance weight per feature map.
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)  # ReLU: keep positive contributions.
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()


def overlay_heatmap(original_rgb: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    """Blend a Grad-CAM heatmap over the original RGB image.

    ``original_rgb`` is a uint8 ``(H, W, 3)`` image in RGB order. Returns an
    RGB uint8 image of the same size with the heatmap overlaid.
    """
    h, w = original_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    overlaid = cv2.addWeighted(original_rgb, 1 - alpha, colored, alpha, 0)
    return overlaid
