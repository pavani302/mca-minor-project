"""Single-image inference + Grad-CAM, shared by the CLI and the web app.

This module is the bridge between a raw uploaded image and everything the
UI needs to display: the predicted label, the confidence, the class
probabilities, and the Grad-CAM overlay.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from . import config, gradcam


@lru_cache(maxsize=1)
def load_model(model_path: str | None = None) -> tf.keras.Model:
    """Load (and cache) the trained Keras model."""
    path = Path(model_path) if model_path else config.MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found at {path}. Run `python -m src.train` first."
        )
    return tf.keras.models.load_model(path)


def load_image(source) -> np.ndarray:
    """Load any image source into a uint8 RGB array at the model input size.

    ``source`` may be a file path, a PIL image, or a file-like object (e.g.
    a Streamlit upload).
    """
    if isinstance(source, Image.Image):
        img = source
    else:
        img = Image.open(source)
    img = img.convert("RGB").resize(config.IMG_SIZE)
    return np.asarray(img, dtype=np.uint8)


def predict(source, model: tf.keras.Model | None = None) -> dict:
    """Run inference on one image and return a rich result dict.

    Returns keys:
        label, confidence, prob_pneumonia, prob_normal,
        original (uint8 RGB), gradcam (uint8 RGB overlay)
    """
    model = model or load_model()

    original = load_image(source)                       # (224, 224, 3) uint8
    batch = tf.keras.applications.densenet.preprocess_input(
        original.astype("float32")[np.newaxis, ...]
    )

    prob_pneumonia = float(model.predict(batch, verbose=0)[0][0])
    prob_normal = 1.0 - prob_pneumonia

    is_pneumonia = prob_pneumonia >= config.DECISION_THRESHOLD
    label = config.POSITIVE_CLASS if is_pneumonia else "NORMAL"
    confidence = prob_pneumonia if is_pneumonia else prob_normal

    # Grad-CAM overlay explaining the decision.
    heatmap = gradcam.compute_heatmap(model, batch[0])
    overlay = gradcam.overlay_heatmap(original, heatmap)

    return {
        "label": label,
        "confidence": confidence,
        "prob_pneumonia": prob_pneumonia,
        "prob_normal": prob_normal,
        "original": original,
        "gradcam": overlay,
    }


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Predict pneumonia for one X-ray image.")
    parser.add_argument("image", help="Path to a chest X-ray image.")
    args = parser.parse_args()

    result = predict(args.image)
    print(f"Prediction : {result['label']}")
    print(f"Confidence : {result['confidence'] * 100:.1f}%")
    print(f"P(pneumonia): {result['prob_pneumonia']:.4f}")


if __name__ == "__main__":
    _cli()
