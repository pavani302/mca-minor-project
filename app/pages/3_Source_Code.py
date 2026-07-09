"""Page 3 — Source Code: an in-app browser for the whole project."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _shared
import streamlit as st

_shared.page_config("Source Code")
from src import config  # noqa: E402

ROOT = config.ROOT_DIR

# Files worth browsing, grouped, each with a one-line description.
FILE_GROUPS: dict[str, list[tuple[str, str]]] = {
    "Machine learning (src/)": [
        ("src/config.py", "All paths & hyperparameters — the single source of truth."),
        ("src/download_data.py", "Download a balanced subset of the chest X-ray dataset."),
        ("src/data_loader.py", "tf.data input pipelines + augmentation + class weights."),
        ("src/model.py", "DenseNet121 transfer-learning model definition."),
        ("src/gradcam.py", "Grad-CAM heatmap computation and overlay."),
        ("src/train.py", "Two-phase training entry point."),
        ("src/evaluate.py", "Metrics + confusion matrix, ROC and Grad-CAM figures."),
        ("src/predict.py", "Single-image inference used by the CLI and this app."),
    ],
    "Web app (app/)": [
        ("app/Home.py", "Landing page — overview, metrics and navigation."),
        ("app/_shared.py", "Shared helpers: model loading, metrics, styling."),
        ("app/pages/1_Try_It.py", "Upload an X-ray → prediction + Grad-CAM."),
        ("app/pages/2_How_It_Works.py", "Visual explanation of the whole project."),
        ("app/pages/3_Source_Code.py", "This source-code browser."),
    ],
    "Project files": [
        ("README.md", "Project overview, setup and usage."),
        ("requirements.txt", "Python dependencies."),
    ],
}

LANG = {".py": "python", ".md": "markdown", ".txt": "text"}

st.title("💻 Source Code Explorer")
st.write(
    "Browse the complete project source code without leaving the app. "
    "Pick a file on the left to view it with syntax highlighting."
)

# Flatten for descriptions lookup.
descriptions = {rel: desc for group in FILE_GROUPS.values() for rel, desc in group}

with st.sidebar:
    st.header("Files")
    selected = None
    for group, files in FILE_GROUPS.items():
        st.caption(group)
        for rel, _desc in files:
            if st.button(rel, key=rel, use_container_width=True):
                st.session_state["selected_file"] = rel

selected = st.session_state.get("selected_file", "src/model.py")

st.subheader(f"`{selected}`")
st.caption(descriptions.get(selected, ""))

path = ROOT / selected
if path.exists():
    text = path.read_text(encoding="utf-8")
    lang = LANG.get(path.suffix, "text")
    n_lines = text.count("\n") + 1
    st.caption(f"{n_lines} lines · {len(text):,} characters")
    st.download_button("Download this file", text, file_name=path.name)
    st.code(text, language=lang, line_numbers=True)
else:
    st.error(f"File not found: {selected}")

st.divider()
st.caption(
    "Tip: the full repository (including training logs and figures) lives in "
    "the project folder and is version-controlled with git."
)
