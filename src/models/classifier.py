from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class SmallClassifier(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

    @torch.no_grad()
    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
