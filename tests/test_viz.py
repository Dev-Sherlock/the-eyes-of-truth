from __future__ import annotations

import numpy as np
import torch
from src.visualization.umap_viz import compute_umap


def test_umap_small_input():
    feats = np.random.rand(4, 16).astype(np.float32)
    emb = compute_umap(feats, n_components=2)
    assert emb.shape == (4, 2)
