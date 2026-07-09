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


def _find_base_and_layer(model: tf.keras.Model):
    """Return the nested DenseNet base model and its target conv layer."""
    base = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base = layer
            break
    if base is None:
        raise RuntimeError("Could not locate the DenseNet base model inside the classifier.")
    return base


def compute_heatmap(model: tf.keras.Model, preprocessed_image: np.ndarray,
                    layer_name: str = config.GRADCAM_LAYER) -> np.ndarray:
    """Compute a normalised Grad-CAM heatmap (values in [0, 1]).

    ``preprocessed_image`` must be a single image already run through the
    DenseNet ``preprocess_input`` step, shaped ``(H, W, 3)`` or ``(1, H, W, 3)``.
    """
    if preprocessed_image.ndim == 3:
        preprocessed_image = preprocessed_image[np.newaxis, ...]

    base = _find_base_and_layer(model)
    target_layer = base.get_layer(layer_name)

    # A model that maps the input image to (conv feature maps, final prediction).
    # We wire the nested base's target layer output up to the top classifier.
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[base.get_layer(layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_output, prediction = grad_model(preprocessed_image, training=False)
        # Binary sigmoid output — gradient of P(pneumonia) w.r.t. feature maps.
        class_channel = prediction[:, 0]

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
