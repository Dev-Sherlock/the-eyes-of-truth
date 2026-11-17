import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app._path  # noqa: F401  -- project-root path bootstrap

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import torch
from configs.settings import get_settings
from src.data.cifar10 import get_cifar10_dataloaders, CLASSES
from src.models.classifier import SmallClassifier
from src.visualization.umap_viz import extract_embeddings, compute_umap

st.title("Embeddings (UMAP)")
s = get_settings()
try:
    ckpt = torch.load(s.model_path, map_location=s.device)
    classes = ckpt.get("classes", CLASSES)
    model = SmallClassifier(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(s.device).eval()
    bundle = get_cifar10_dataloaders(s.data_dir, batch_size=s.batch_size, num_workers=0, image_size=s.image_size, limit=512)
    feats, labels = extract_embeddings(model, bundle.test, device=s.device, max_batches=10)
    emb = compute_umap(feats)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10", s=8, alpha=0.7)
    ax.set_title("UMAP of classifier backbone features")
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Embeddings unavailable: {e}")
