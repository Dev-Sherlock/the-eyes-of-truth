from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DINOHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, hidden_dim: int = 512, bottleneck_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class DINO(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        student_backbone = models.resnet18(weights=None)
        teacher_backbone = models.resnet18(weights=None)
        embed_dim = student_backbone.fc.in_features
        student_backbone.fc = nn.Identity()
        teacher_backbone.fc = nn.Identity()
        self.student_backbone = student_backbone
        self.teacher_backbone = teacher_backbone
        self.student_head = DINOHead(embed_dim, out_dim=out_dim)
        self.teacher_head = DINOHead(embed_dim, out_dim=out_dim)
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

    def forward_student(self, x):
        return self.student_head(self.student_backbone(x))

    @torch.no_grad()
    def forward_teacher(self, x):
        return self.teacher_head(self.teacher_backbone(x))

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        for ps, pt in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)
        for ps, pt in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)


def dino_loss(student_outs, teacher_outs, student_temp=0.1, teacher_temp=0.04):
    total = 0.0
    n = 0
    teacher_probs = [F.softmax(t / teacher_temp, dim=-1).detach() for t in teacher_outs]
    for iq, t in enumerate(teacher_probs):
        for is_, s in enumerate(student_outs):
            if is_ == iq:
                continue
            s_log = F.log_softmax(s / student_temp, dim=-1)
            total += -(t * s_log).sum(dim=-1).mean()
            n += 1
    return total / max(n, 1)
