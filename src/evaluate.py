"""Evaluate the trained model and generate report-ready figures.

Produces:
    * a metrics dict (accuracy, precision, recall, F1, AUC, per-class support)
    * confusion_matrix.png
    * roc_curve.png
    * sample_gradcams.png  (a grid of test images with Grad-CAM overlays)

Run standalone (loads the saved model)::

    python -m src.evaluate
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from . import config, gradcam


def _collect_predictions(model: tf.keras.Model, test_ds: tf.data.Dataset):
    """Return (y_true, y_prob) over the whole test dataset."""
    y_true, y_prob = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0).ravel()
        y_prob.extend(probs.tolist())
        y_true.extend(labels.numpy().ravel().tolist())
    return np.array(y_true, dtype=int), np.array(y_prob, dtype=float)


def _plot_confusion_matrix(y_true, y_pred, path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_roc(y_true, y_prob, path) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, lw=2, label=f"DenseNet121 (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (test set)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return float(roc_auc)


def _plot_sample_gradcams(model: tf.keras.Model, path, n: int = 6) -> None:
    """Grid of test images with their Grad-CAM overlays and predictions."""
    from . import predict as predict_mod

    # Grab a few images per class from the test split on disk.
    test_dir = config.RAW_DATA_DIR / "test"
    paths = []
    for label_name in config.CLASS_NAMES:
        cls_dir = test_dir / label_name
        if cls_dir.exists():
            paths.extend(sorted(cls_dir.glob("*.jpeg"))[: n // 2])
    paths = paths[:n]
    if not paths:
        return

    cols = 3
    rows = (len(paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, img_path in zip(axes, paths):
        res = predict_mod.predict(str(img_path), model=model)
        ax.imshow(res["gradcam"])
        truth = img_path.parent.name
        ax.set_title(
            f"True: {truth}\nPred: {res['label']} ({res['confidence']*100:.0f}%)",
            fontsize=10,
        )
        ax.axis("off")
    for ax in axes[len(paths):]:
        ax.axis("off")

    fig.suptitle("Grad-CAM explanations on test X-rays", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def evaluate_model(model: tf.keras.Model, test_ds: tf.data.Dataset) -> dict:
    """Compute metrics and write all figures to ``reports/figures``."""
    config.ensure_dirs()

    y_true, y_prob = _collect_predictions(model, test_ds)
    y_pred = (y_prob >= config.DECISION_THRESHOLD).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_test": int(len(y_true)),
        "n_normal": int((y_true == 0).sum()),
        "n_pneumonia": int((y_true == 1).sum()),
        "positive_class": config.POSITIVE_CLASS,
    }

    _plot_confusion_matrix(y_true, y_pred, config.FIGURES_DIR / "confusion_matrix.png")
    metrics["auc"] = _plot_roc(y_true, y_prob, config.FIGURES_DIR / "roc_curve.png")
    _plot_sample_gradcams(model, config.FIGURES_DIR / "sample_gradcams.png")

    return metrics


def main() -> None:
    from . import data_loader, predict as predict_mod

    model = predict_mod.load_model()
    _, _, test_ds = data_loader.get_datasets()
    metrics = evaluate_model(model, test_ds)
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
