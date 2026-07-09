"""DenseNet121 transfer-learning model for pneumonia detection.

The model is a standard transfer-learning head on top of an ImageNet
pretrained DenseNet121 backbone::

    DenseNet121 (frozen/fine-tuned)
        -> GlobalAveragePooling2D
        -> Dropout
        -> Dense(1, sigmoid)   # P(PNEUMONIA)
"""
from __future__ import annotations

import tensorflow as tf

from . import config


def build_model(dropout: float = 0.3) -> tf.keras.Model:
    """Build the DenseNet121 classifier with the base initially frozen."""
    base = tf.keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=(*config.IMG_SIZE, config.IMG_CHANNELS),
    )
    base.trainable = False  # Phase 1: freeze the backbone.

    inputs = tf.keras.Input(shape=(*config.IMG_SIZE, config.IMG_CHANNELS))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = tf.keras.Model(inputs, outputs, name="pneumonia_densenet121")
    return model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    """Compile with binary cross-entropy and the metrics we report on."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


def unfreeze_top(model: tf.keras.Model, finetune_at: int = config.FINETUNE_AT) -> None:
    """Unfreeze the top layers of the DenseNet backbone for fine-tuning.

    BatchNormalization layers are kept frozen (running statistics stay put),
    which is the standard recipe for stable fine-tuning.
    """
    # The DenseNet121 backbone is the first non-input layer in the model.
    base = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base = layer
            break
    if base is None:
        raise RuntimeError("Could not locate the DenseNet base model.")

    base.trainable = True
    for i, layer in enumerate(base.layers):
        if i < finetune_at or isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
