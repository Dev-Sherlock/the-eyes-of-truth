import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import streamlit as st
import torch

import app._path  # noqa: F401  -- project-root path bootstrap
from configs.settings import get_settings
from src.data.cifar10 import CLASSES, get_cifar10_dataloaders
from src.evaluation.metrics import evaluate_classifier
from src.evaluation.robustness import robustness_check
from src.models.classifier import SmallClassifier

st.title("Evaluation")
s = get_settings()
try:
    ckpt = torch.load(s.model_path, map_location=s.device)
    model = SmallClassifier(num_classes=len(ckpt.get("classes", CLASSES)))
    model.load_state_dict(ckpt["model_state"])
    model.to(s.device).eval()
    bundle = get_cifar10_dataloaders(
        s.data_dir, batch_size=s.batch_size, num_workers=0, image_size=s.image_size, limit=256
    )
    metrics = evaluate_classifier(model, bundle.test, device=s.device)
    robust = robustness_check(model, bundle.test, device=s.device)
    st.subheader("Metrics")
    st.json({k: v for k, v in metrics.items() if k != "confusion_matrix"})
    st.subheader("Robustness (Gaussian noise)")
    st.json(robust)
    st.subheader("Confusion Matrix")
    st.write(metrics.get("confusion_matrix", []))
except Exception as e:  # noqa: BLE001
    st.warning(f"Evaluation unavailable: {e}. Train/fine-tune first to create {s.model_path}.")
