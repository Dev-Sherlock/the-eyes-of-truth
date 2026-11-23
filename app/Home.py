# ruff: noqa: N999
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

import app._path  # noqa: F401  -- project-root path bootstrap

st.set_page_config(page_title="The Eyes of Truth", layout="wide")
st.title("The Eyes of Truth")
st.write("DINO self-supervised learning dashboard for CIFAR-10.")
st.markdown("""
**Pages**
1. Overview
2. Training
3. Data Quality
4. Evaluation
5. Embeddings
6. Explainability
7. Inference
""")
