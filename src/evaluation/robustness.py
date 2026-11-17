from __future__ import annotations

import random
import torch
from torchvision import transforms


def add_noise(img_tensor: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    return torch.clamp(img_tensor + torch.randn_like(img_tensor) * std, 0.0, 1.0)


@torch.no_grad()
def robustness_check(model, loader, device="cpu", noise_std: float = 0.05):
    model.eval()
    clean_correct = noisy_correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred_clean = model(x).argmax(dim=1)
        pred_noisy = model(add_noise(x, noise_std)).argmax(dim=1)
        clean_correct += (pred_clean == y).sum().item()
        noisy_correct += (pred_noisy == y).sum().item()
        total += y.size(0)
    total = max(total, 1)
    return {"clean_acc": clean_correct / total, "noisy_acc": noisy_correct / total, "noise_std": noise_std}
