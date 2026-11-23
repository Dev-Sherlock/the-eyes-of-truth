import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

import app._path  # noqa: F401  -- project-root path bootstrap

st.title("Overview")
st.write("Project: DINO (self-supervised) on CIFAR-10, CPU-friendly demo.")
st.write(
    "Use the sidebar pages for training status, quality, evaluation, embeddings, explainability, and inference."
)
