from __future__ import annotations

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


@torch.no_grad()
def evaluate_classifier(model, loader, device="cpu"):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(y.tolist())
    acc = accuracy_score(targets, preds) if targets else 0.0
    p, r, f1, _ = precision_recall_fscore_support(targets, preds, average="macro", zero_division=0)
    cm = confusion_matrix(targets, preds) if targets else []
    return {"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1), "confusion_matrix": cm.tolist() if hasattr(cm, "tolist") else cm}
