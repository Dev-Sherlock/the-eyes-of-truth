from __future__ import annotations

import argparse
import os

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

from configs.settings import get_settings
from src.training.train import finetune_classifier

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default=None)
    args = p.parse_args()
    path = finetune_classifier(get_settings(), backbone_path=args.backbone)
    print(f"saved classifier to {path}")
