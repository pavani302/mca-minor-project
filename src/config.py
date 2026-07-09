"""Central configuration for the pneumonia-detection project.

All paths, hyperparameters, and shared constants live here so that the
training scripts, evaluation code and the Streamlit app all read from a
single source of truth.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"        # full dataset (git-ignored)
SAMPLE_DATA_DIR = DATA_DIR / "sample"  # tiny committed sample for the demo

MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Where the trained model + its metadata are written / read from.
MODEL_PATH = MODELS_DIR / "pneumonia_densenet121.keras"
METRICS_PATH = MODELS_DIR / "metrics.json"

# --------------------------------------------------------------------------- #
# Image / model settings
# --------------------------------------------------------------------------- #
IMG_SIZE = (224, 224)          # DenseNet121 default input size
IMG_CHANNELS = 3               # X-rays are converted to 3-channel RGB
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]   # index 0 / 1 -> label
POSITIVE_CLASS = "PNEUMONIA"
DECISION_THRESHOLD = 0.5       # sigmoid output >= threshold -> PNEUMONIA

# Last convolutional layer of DenseNet121 — used by Grad-CAM.
GRADCAM_LAYER = "conv5_block16_concat"

# --------------------------------------------------------------------------- #
# Training hyperparameters
# --------------------------------------------------------------------------- #
BATCH_SIZE = 32
SEED = 42

# Phase 1: train the classification head with the base frozen.
HEAD_EPOCHS = 8
HEAD_LR = 1e-3

# Phase 2: unfreeze the top of the base and fine-tune with a low LR.
FINETUNE_EPOCHS = 6
FINETUNE_LR = 1e-5
FINETUNE_AT = 300  # unfreeze layers from this index onward (DenseNet121 ~427 layers)

# --------------------------------------------------------------------------- #
# Dataset download (small subset for a fast, real baseline)
# --------------------------------------------------------------------------- #
# Public HuggingFace mirror of the Kaggle Chest X-Ray Pneumonia dataset.
# No Kaggle login required.
HF_DATASET_ID = "hf-vision/chest-xray-pneumonia"
# Images per class to pull for the quick subset baseline.
SUBSET_PER_CLASS_TRAIN = 400
SUBSET_PER_CLASS_VAL = 60
SUBSET_PER_CLASS_TEST = 100


def ensure_dirs() -> None:
    """Create the output directories if they do not yet exist."""
    for d in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR, RAW_DATA_DIR, SAMPLE_DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)
