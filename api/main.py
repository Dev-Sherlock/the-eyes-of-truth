from __future__ import annotations

import io

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from configs.settings import get_settings
from src.data.cifar10 import CLASSES, base_transform
from src.models.classifier import SmallClassifier

app = FastAPI(title="Eyes of Truth API")


def _load_model():
    s = get_settings()
    device = torch.device(s.device)
    try:
        ckpt = torch.load(s.model_path, map_location=device)
    except (OSError, RuntimeError) as e:
        raise RuntimeError(f"Model not found or unreadable at {s.model_path}: {e}")
    classes = tuple(ckpt.get("classes", CLASSES))
    model = SmallClassifier(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, classes, s, device


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):  # noqa: B008
    try:
        model, classes, s, device = _load_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except (OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid image file")
    x = base_transform(s.image_size)(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    idx = int(probs.argmax().item())
    return {
        "class": classes[idx],
        "class_index": idx,
        "confidence": float(probs[idx].item()),
        "probabilities": {classes[i]: float(probs[i].item()) for i in range(len(classes))},
    }
