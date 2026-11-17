import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app._path  # noqa: F401  -- project-root path bootstrap

import io
import streamlit as st
import torch
from PIL import Image
from configs.settings import get_settings
from src.data.cifar10 import base_transform, CLASSES
from src.models.classifier import SmallClassifier

st.title("Inference")
s = get_settings()
upload = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
if upload:
    img = Image.open(upload).convert("RGB")
    st.image(img, caption="Input", width=256)
    try:
        ckpt = torch.load(s.model_path, map_location=s.device)
        classes = tuple(ckpt.get("classes", CLASSES))
        model = SmallClassifier(num_classes=len(classes))
        model.load_state_dict(ckpt["model_state"])
        model.to(s.device).eval()
        x = base_transform(s.image_size)(img).unsqueeze(0).to(s.device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
        idx = int(probs.argmax().item())
        st.success(f"Prediction: {classes[idx]} ({probs[idx].item():.3f})")
        st.bar_chart({classes[i]: float(probs[i].item()) for i in range(len(classes))})
    except Exception as e:
        st.warning(f"Inference unavailable: {e}. Expected model at {s.model_path}")
else:
    st.info("Upload an image to run prediction.")
