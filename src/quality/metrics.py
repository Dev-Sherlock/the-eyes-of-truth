from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def blur_score(image_path: str) -> float:
    """Simple variance-of-Laplacian-like proxy without cv2: uses gradient magnitude variance."""
    img = Image.open(image_path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    gy, gx = np.gradient(arr)
    mag = np.sqrt(gx**2 + gy**2)
    return float(mag.var())


def dataset_summary(data_dir: str) -> dict:
    root = Path(data_dir)
    count = sum(1 for _ in root.rglob("*.png")) if root.exists() else 0
    return {"path": str(root), "png_files": count}
