"""Build ``tf.data`` input pipelines for training and evaluation.

Images are loaded from an ``ImageFolder`` style directory (see
``download_data.py``), resized to the DenseNet121 input size, and passed
through the DenseNet ``preprocess_input`` function. Augmentation (flip,
rotation, zoom, contrast) is applied to the training split only.
"""
from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from . import config

AUTOTUNE = tf.data.AUTOTUNE


def _make_dataset(directory: Path, shuffle: bool) -> tf.data.Dataset:
    """Create a labelled dataset from a split directory."""
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",              # single sigmoid output
        class_names=config.CLASS_NAMES,   # fix label order: NORMAL=0, PNEUMONIA=1
        color_mode="rgb",
        batch_size=config.BATCH_SIZE,
        image_size=config.IMG_SIZE,
        shuffle=shuffle,
        seed=config.SEED,
    )


# Augmentation applied on-the-fly to the training images.
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ],
    name="data_augmentation",
)


def _preprocess(images: tf.Tensor) -> tf.Tensor:
    """Scale pixels the way DenseNet121 expects."""
    return tf.keras.applications.densenet.preprocess_input(images)


def get_datasets(data_dir: Path | None = None):
    """Return ``(train_ds, val_ds, test_ds)`` ready for ``model.fit``.

    Augmentation is baked into the training pipeline; all splits are
    preprocessed with the DenseNet normalisation and prefetched.
    """
    data_dir = Path(data_dir) if data_dir else config.RAW_DATA_DIR

    train_ds = _make_dataset(data_dir / "train", shuffle=True)
    val_ds = _make_dataset(data_dir / "val", shuffle=False)
    test_ds = _make_dataset(data_dir / "test", shuffle=False)

    train_ds = (
        train_ds
        .map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
        .map(lambda x, y: (_preprocess(x), y), num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )
    val_ds = val_ds.map(lambda x, y: (_preprocess(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (_preprocess(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds


def compute_class_weights(data_dir: Path | None = None) -> dict[int, float]:
    """Inverse-frequency class weights to counter dataset imbalance.

    The Kaggle dataset has far more pneumonia than normal images, so we
    up-weight the minority class during training.
    """
    data_dir = Path(data_dir) if data_dir else config.RAW_DATA_DIR
    train_dir = data_dir / "train"

    counts = []
    for label_name in config.CLASS_NAMES:
        n = len(list((train_dir / label_name).glob("*")))
        counts.append(max(n, 1))

    total = sum(counts)
    n_classes = len(counts)
    # weight = total / (n_classes * count_for_class)
    return {i: total / (n_classes * c) for i, c in enumerate(counts)}
