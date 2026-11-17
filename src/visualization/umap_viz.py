from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def extract_embeddings(model, loader, device="cpu", max_batches: int = 20):
    model.eval()
    feats, labels = [], []
    backbone = getattr(model, "backbone", model)
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        f = backbone(x)
        if f.ndim > 2:
            f = torch.flatten(f, 1)
        feats.append(f.cpu().numpy())
        labels.append(y.numpy())
    if not feats:
        return np.empty((0, 0)), np.empty((0,), dtype=int)
    return np.concatenate(feats), np.concatenate(labels)


def compute_umap(features: np.ndarray, n_components: int = 2):
    try:
        import umap
    except Exception as e:
        raise RuntimeError("umap-learn is required for embeddings visualization") from e
    if len(features) < 5:
        return np.zeros((len(features), n_components))
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    return reducer.fit_transform(features)
