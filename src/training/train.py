from __future__ import annotations

import os
from pathlib import Path

import mlflow
import torch
from torch import nn
from torch.optim import AdamW

from configs.settings import get_settings
from src.data.cifar10 import get_cifar10_dataloaders, get_dino_dataloader
from src.models.classifier import SmallClassifier
from src.models.dino import DINO, dino_loss


def ensure_dir(path: str | os.PathLike):
    Path(path).mkdir(parents=True, exist_ok=True)


def pretrain_dino(settings=None):
    s = settings or get_settings()
    ensure_dir(s.artifacts_dir)
    device = torch.device(s.device)
    loader = get_dino_dataloader(
        s.data_dir,
        batch_size=s.batch_size,
        num_workers=s.num_workers,
        global_crop_size=s.global_crop_size,
        local_crop_size=s.local_crop_size,
        n_global=s.n_global_crops,
        n_local=s.n_local_crops,
    )
    model = DINO(out_dim=s.out_dim).to(device)
    opt = AdamW(
        list(model.student_backbone.parameters()) + list(model.student_head.parameters()),
        lr=s.dino_lr,
        weight_decay=s.dino_weight_decay,
    )
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    mlflow.set_experiment("dino_pretrain")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "epochs": s.dino_epochs,
                "batch_size": s.batch_size,
                "lr": s.dino_lr,
                "device": s.device,
            }
        )
        for epoch in range(s.dino_epochs):
            model.train()
            running = 0.0
            for crops, _ in loader:
                crops = [c.to(device) for c in crops]
                student_outs = [model.forward_student(c) for c in crops]
                with torch.no_grad():
                    teacher_outs = [model.forward_teacher(c) for c in crops[: s.n_global_crops]]
                loss = dino_loss(student_outs, teacher_outs)
                opt.zero_grad()
                loss.backward()
                opt.step()
                model.update_teacher(s.teacher_momentum)
                running += loss.item()
            avg = running / max(len(loader), 1)
            mlflow.log_metric("train_loss", avg, step=epoch)
            print(f"epoch {epoch+1}/{s.dino_epochs} loss={avg:.4f}")
        torch.save(model.student_backbone.state_dict(), s.backbone_path)
    return s.backbone_path


def finetune_classifier(settings=None, backbone_path: str | None = None):
    s = settings or get_settings()
    ensure_dir(s.artifacts_dir)
    device = torch.device(s.device)
    bundle = get_cifar10_dataloaders(
        s.data_dir, batch_size=s.batch_size, num_workers=s.num_workers, image_size=s.image_size
    )
    model = SmallClassifier(num_classes=s.num_classes).to(device)
    bp = backbone_path or s.backbone_path
    if os.path.exists(bp):
        state = torch.load(bp, map_location=device)
        model.backbone.load_state_dict(state, strict=False)
    criterion = nn.CrossEntropyLoss()
    opt = AdamW(model.parameters(), lr=s.finetune_lr)
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    mlflow.set_experiment("classifier_finetune")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "epochs": s.finetune_epochs,
                "batch_size": s.batch_size,
                "lr": s.finetune_lr,
                "device": s.device,
            }
        )
        for epoch in range(s.finetune_epochs):
            model.train()
            correct = total = 0
            running = 0.0
            for x, y in bundle.train:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                opt.step()
                running += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
            acc = correct / max(total, 1)
            avg = running / max(len(bundle.train), 1)
            mlflow.log_metrics({"train_loss": avg, "train_acc": acc}, step=epoch)
            print(f"finetune epoch {epoch+1}/{s.finetune_epochs} loss={avg:.4f} acc={acc:.4f}")
        torch.save({"model_state": model.state_dict(), "classes": bundle.classes}, s.model_path)
    return s.model_path
