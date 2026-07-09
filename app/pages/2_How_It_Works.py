"""Page 2 — How It Works: a visual explanation of the whole project."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _shared
import streamlit as st

_shared.page_config("How It Works")
from src import config  # noqa: E402

FIG = config.FIGURES_DIR

st.title("📊 How It Works")
st.write("A visual walk-through of the pipeline, model, and results.")

# ------------------------------------------------------------ pipeline
st.header("1 · The pipeline")
st.markdown(
    """
```
Chest X-ray  ─▶  Preprocess          ─▶  DenseNet121        ─▶  Sigmoid    ─▶  NORMAL / PNEUMONIA
                 (resize 224×224,        (ImageNet backbone,     (0 → 1)        + confidence
                  RGB, normalise)         fine-tuned)                           + Grad-CAM
```
"""
)
st.markdown(
    """
1. **Preprocess** — every X-ray is resized to 224×224, converted to 3-channel
   RGB, and normalised with DenseNet's `preprocess_input`.
2. **Augment (training only)** — random flips, rotation, zoom and contrast make
   the model robust and reduce overfitting.
3. **DenseNet121** — a densely-connected CNN pretrained on ImageNet extracts
   rich visual features; we fine-tune it for X-rays.
4. **Classify** — a small head (global pooling → dropout → 1 sigmoid unit)
   outputs the probability of pneumonia.
5. **Explain** — Grad-CAM turns the model's internal gradients into a heatmap
   over the X-ray.
"""
)

# ------------------------------------------------------------ dataset
st.header("2 · The dataset")
metrics = _shared.load_metrics()
col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
- **Source:** Chest X-Ray Images (Pneumonia) — Kaggle / Kermany et al.
- **Classes:** `NORMAL` and `PNEUMONIA`
- **Split:** train / validation / test
- **Imbalance:** more pneumonia than normal images, handled with
  **class weights** during training.
"""
    )
with col2:
    if metrics:
        st.markdown(
            f"""
**Test set used for evaluation**

| Class | Images |
|---|---|
| Normal | {metrics['n_normal']} |
| Pneumonia | {metrics['n_pneumonia']} |
| **Total** | **{metrics['n_test']}** |
"""
        )

# ------------------------------------------------------------ architecture
st.header("3 · The model — DenseNet121 transfer learning")
st.markdown(
    """
Instead of training a CNN from scratch (which needs huge data), we **transfer
learn** from **DenseNet121** pretrained on ImageNet. Training happens in two
phases:

- **Phase 1 — feature extraction:** freeze the DenseNet backbone and train only
  the new classification head.
- **Phase 2 — fine-tuning:** unfreeze the top layers of the backbone and
  continue training with a very small learning rate so the pretrained features
  adapt to X-rays without being destroyed.

Regularisation: **dropout**, **data augmentation**, **early stopping** and
**learning-rate reduction** together prevent overfitting.
"""
)

# ------------------------------------------------------------ results
st.header("4 · Results")
if metrics:
    _shared.metric_badges(metrics)
    st.caption(
        "For a medical screening tool, **recall** (catching real pneumonia "
        "cases) is as important as raw accuracy — a missed case is worse than "
        "a false alarm."
    )

    def show(fig_name: str, caption: str):
        p = FIG / fig_name
        if p.exists():
            st.image(str(p), caption=caption, use_column_width=True)
        else:
            st.info(f"`{fig_name}` will appear here after training.")

    c1, c2 = st.columns(2)
    with c1:
        show("training_history.png", "Training & validation curves across both phases")
        show("confusion_matrix.png", "Confusion matrix on the test set")
    with c2:
        show("roc_curve.png", "ROC curve (higher AUC = better separation)")
        show("sample_gradcams.png", "Grad-CAM explanations on sample test X-rays")
else:
    _shared.model_status_banner()
    st.markdown(
        "Once you run `python -m src.train`, this section fills with the "
        "training curves, confusion matrix, ROC curve and Grad-CAM examples."
    )

# ------------------------------------------------------------ gradcam
st.header("5 · Why Grad-CAM matters")
st.markdown(
    """
A prediction alone doesn't tell you **why**. **Grad-CAM** (Gradient-weighted
Class Activation Mapping) computes how much each region of the final
convolutional feature map contributed to the pneumonia score, producing a
heatmap over the X-ray. This lets a clinician confirm the model is looking at
**clinically relevant lung regions** — turning a black box into a transparent,
trustworthy aid.
"""
)

st.divider()
st.info(_shared.DISCLAIMER, icon="⚠️")
