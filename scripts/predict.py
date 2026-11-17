from __future__ import annotations

import argparse
import json
import torch
from PIL import Image
from configs.settings import get_settings
from src.data.cifar10 import base_transform, CLASSES
from src.models.classifier import SmallClassifier


def predict_image(image_path: str, settings=None):
    s = settings or get_settings()
    device = torch.device(s.device)
    ckpt = torch.load(s.model_path, map_location=device)
    classes = tuple(ckpt.get("classes", CLASSES))
    model = SmallClassifier(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    img = Image.open(image_path).convert("RGB")
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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    print(json.dumps(predict_image(args.image), indent=2))
