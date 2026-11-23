import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st
import torch
from PIL import Image

import app._path  # noqa: F401  -- project-root path bootstrap
from configs.settings import get_settings
from src.data.cifar10 import CLASSES, base_transform
from src.models.classifier import SmallClassifier
from src.visualization.gradcam import GradCAM

st.title("Explainability (Grad-CAM)")
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
        # target last conv block in resnet layer4
        cam_engine = GradCAM(model, model.backbone.layer4[-1])
        cam, cls_idx = cam_engine(x)
        cam_engine.close()
        cam_np = cam.numpy()
        heat = (
            (np.stack([cam_np, np.zeros_like(cam_np), 1 - cam_np], axis=-1) * 255)
            .clip(0, 255)
            .astype(np.uint8)
        )
        heat_img = Image.fromarray(heat).resize(img.size)
        overlay = Image.blend(img.resize(heat_img.size), heat_img, alpha=0.45)
        st.image(overlay, caption=f"Grad-CAM: {classes[cls_idx]}", width=256)
    except Exception as e:  # noqa: BLE001
        st.warning(f"Explainability unavailable: {e}")
else:
    st.info("Upload an image and ensure a trained classifier exists.")
