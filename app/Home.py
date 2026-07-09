"""Streamlit entry point — project overview & navigation.

Run the app from the project root with::

    streamlit run app/Home.py
"""
from __future__ import annotations

import _shared
import streamlit as st

_shared.page_config("Home")

st.title("🫁 Pneumonia Detection from Chest X-Rays")
st.markdown(
    "#### A deep-learning diagnostic aid using DenseNet121 transfer learning "
    "with Grad-CAM explainability"
)

st.markdown(
    """
This is an **MCA minor project** that detects **pneumonia** in chest X-ray
images using a convolutional neural network. A pretrained **DenseNet121** is
fine-tuned on a chest X-ray dataset, and **Grad-CAM** highlights the lung
regions that drive each prediction — so the AI's decision is *interpretable*,
not a black box.
"""
)

# Headline metrics, if the model has been trained.
metrics = _shared.load_metrics()
if metrics:
    st.subheader("Model performance on the held-out test set")
    _shared.metric_badges(metrics)
    st.caption(
        f"Measured on {metrics['n_test']} unseen test images "
        f"({metrics['n_normal']} normal, {metrics['n_pneumonia']} pneumonia)."
    )
else:
    _shared.model_status_banner()

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 🔬 Try It")
    st.write("Upload a chest X-ray and get an instant prediction with a "
             "confidence score and a Grad-CAM heatmap.")
    st.page_link("pages/1_Try_It.py", label="Open the detector", icon="🔬")
with col2:
    st.markdown("### 📊 How It Works")
    st.write("A visual walk-through of the dataset, the DenseNet121 "
             "architecture, training, and the evaluation results.")
    st.page_link("pages/2_How_It_Works.py", label="See the explanation", icon="📊")
with col3:
    st.markdown("### 💻 Source Code")
    st.write("Browse the full project source code — data pipeline, model, "
             "Grad-CAM, training and this app — right here.")
    st.page_link("pages/3_Source_Code.py", label="Explore the code", icon="💻")

st.divider()

with st.expander("About this project", expanded=False):
    st.markdown(
        """
- **Task:** Binary image classification — *Normal* vs *Pneumonia*
- **Model:** DenseNet121 (ImageNet pretrained) + custom classification head,
  fine-tuned in two phases
- **Explainability:** Grad-CAM (Gradient-weighted Class Activation Mapping)
- **Stack:** Python · TensorFlow/Keras · OpenCV · scikit-learn · Streamlit
- **Dataset:** Chest X-Ray Pneumonia (Kaggle / Kermany et al.)

**Objectives (from the proposal)**
1. A deep-learning system for pneumonia detection from chest X-rays.
2. Pretrained CNN models to improve classification accuracy.
3. Explainable-AI techniques for interpreting the results.
4. A simple interface to demonstrate the model's predictions.
"""
    )

st.info(_shared.DISCLAIMER, icon="⚠️")
