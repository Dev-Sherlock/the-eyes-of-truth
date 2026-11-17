from __future__ import annotations

import os

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

from configs.settings import get_settings
from src.training.train import pretrain_dino

if __name__ == "__main__":
    path = pretrain_dino(get_settings())
    print(f"saved backbone to {path}")
