import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app._path  # noqa: F401  -- project-root path bootstrap

import streamlit as st
from src.quality.metrics import dataset_summary
from configs.settings import get_settings

st.title("Data Quality")
s = get_settings()
st.json(dataset_summary(s.data_dir))
st.write("Use `src.quality.metrics.blur_score(path)` for a simple blur/sharpness proxy on exported images.")
