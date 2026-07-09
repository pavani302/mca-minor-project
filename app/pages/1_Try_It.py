"""Page 1 — Try It: upload an X-ray, get a prediction + Grad-CAM."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _shared
import streamlit as st

_shared.page_config("Try It")
from src import config  # noqa: E402

st.title("🔬 Try It — Pneumonia Detector")
st.write(
    "Upload a chest X-ray image (JPEG/PNG). The model predicts whether it "
    "shows **pneumonia** or is **normal**, reports its confidence, and shows a "
    "**Grad-CAM heatmap** of the regions that influenced the decision."
)

model = _shared.get_model()
if model is None:
    _shared.model_status_banner()
    st.stop()

# ------------------------------------------------------------------ input
sample_dir = config.SAMPLE_DATA_DIR
sample_files = sorted(sample_dir.glob("*.jpeg")) if sample_dir.exists() else []

uploaded = st.file_uploader("Upload a chest X-ray", type=["jpg", "jpeg", "png"])

chosen_sample = None
if sample_files:
    st.caption("…or pick a bundled example:")
    cols = st.columns(len(sample_files))
    for col, f in zip(cols, sample_files):
        with col:
            st.image(str(f), caption=f.stem, use_container_width=True)
            if st.button("Use this", key=f.name):
                chosen_sample = f

source = uploaded or chosen_sample

# ------------------------------------------------------------------ predict
if source is not None:
    from src import predict as predict_mod

    with st.spinner("Analysing the X-ray…"):
        result = predict_mod.predict(source, model=model)

    st.divider()
    is_pneumonia = result["label"] == config.POSITIVE_CLASS
    verdict_col, conf_col = st.columns([2, 1])
    with verdict_col:
        if is_pneumonia:
            st.error(f"### 🩺 Prediction: PNEUMONIA")
        else:
            st.success(f"### 🩺 Prediction: NORMAL")
    with conf_col:
        st.metric("Confidence", f"{result['confidence']*100:.1f}%")

    # Probability bars for both classes.
    st.write("**Class probabilities**")
    st.write("Pneumonia")
    st.progress(result["prob_pneumonia"], text=f"{result['prob_pneumonia']*100:.1f}%")
    st.write("Normal")
    st.progress(result["prob_normal"], text=f"{result['prob_normal']*100:.1f}%")

    if 0.4 < result["prob_pneumonia"] < 0.6:
        st.warning("The model is **uncertain** about this image (probability "
                   "near the 50% decision boundary). Interpret with caution.",
                   icon="⚖️")

    # Images: original + Grad-CAM.
    st.divider()
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image(result["original"], caption="Uploaded X-ray", use_container_width=True)
    with img_col2:
        st.image(result["gradcam"],
                 caption="Grad-CAM — red/yellow = most influential regions",
                 use_container_width=True)

    with st.expander("How to read the Grad-CAM heatmap"):
        st.write(
            "Grad-CAM overlays a heatmap on the X-ray. **Warm colours "
            "(red/yellow)** mark the areas the model weighted most heavily when "
            "making its prediction. For pneumonia, these often correspond to "
            "regions of lung opacity/consolidation — a sanity check that the "
            "model looks at clinically relevant areas rather than artifacts."
        )
else:
    st.info("Upload an image or choose an example to run a prediction.")

st.divider()
st.info(_shared.DISCLAIMER, icon="⚠️")
