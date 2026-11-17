import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app._path  # noqa: F401  -- project-root path bootstrap

import streamlit as st
from configs.settings import get_settings

st.title("Training")
s = get_settings()
st.code("python -m scripts.pretrain_dino`npython -m scripts.finetune_classifier", language="powershell")
st.json({"dino_epochs": s.dino_epochs, "finetune_epochs": s.finetune_epochs, "batch_size": s.batch_size, "device": s.device})
st.info("Run scripts from the project root with the venv Python. Artifacts are saved under artifacts/.")
