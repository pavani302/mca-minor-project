"""Shared helpers for the Streamlit app (paths, model loading, metrics)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# Make the project root importable so `import src...` works when Streamlit
# runs a page from the app/ directory.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

PRIMARY = "#2563eb"
ACCENT = "#0ea5e9"

DISCLAIMER = (
    "⚠️ **For educational demonstration only.** This tool is a student minor "
    "project and is **not** a medical device. It must not be used for real "
    "diagnosis. Always consult a qualified radiologist or physician."
)


def page_config(title: str) -> None:
    st.set_page_config(
        page_title=f"{title} · Pneumonia Detection",
        page_icon="🫁",
        layout="wide",
    )


@st.cache_resource(show_spinner="Loading the trained model…")
def get_model():
    """Load and cache the trained model, or return None if not trained yet."""
    from src import predict

    try:
        return predict.load_model()
    except FileNotFoundError:
        return None


def load_metrics() -> dict | None:
    """Read the saved test-set metrics, if training has produced them."""
    if config.METRICS_PATH.exists():
        with open(config.METRICS_PATH) as f:
            return json.load(f)
    return None


def model_status_banner() -> bool:
    """Warn if no trained model is present. Returns True if a model exists."""
    if get_model() is None:
        st.warning(
            "No trained model found yet. Run **`python -m src.download_data`** "
            "then **`python -m src.train`** to produce `models/"
            f"{config.MODEL_PATH.name}`. The interface below is fully functional "
            "once a model exists.",
            icon="🔧",
        )
        return False
    return True


def metric_badges(metrics: dict) -> None:
    """Render the headline test-set metrics as Streamlit metric tiles."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Accuracy", f"{metrics['accuracy']*100:.1f}%")
    c2.metric("Recall (Sensitivity)", f"{metrics['recall']*100:.1f}%")
    c3.metric("Precision", f"{metrics['precision']*100:.1f}%")
    c4.metric("AUC", f"{metrics.get('auc', 0)*100:.1f}%")
