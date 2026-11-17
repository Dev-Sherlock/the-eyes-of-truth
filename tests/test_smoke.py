import torch
from configs.settings import get_settings
from src.data.cifar10 import get_cifar10_dataloaders
from src.models.classifier import SmallClassifier


def test_settings_defaults():
    s = get_settings()
    assert s.device == "cpu"
    assert s.num_classes == 10


def test_dataloader_and_model_forward_cpu():
    s = get_settings()
    bundle = get_cifar10_dataloaders(s.data_dir, batch_size=4, num_workers=0, image_size=32, limit=8)
    x, y = next(iter(bundle.train))
    model = SmallClassifier(num_classes=10)
    out = model(x)
    assert out.shape == (x.shape[0], 10)
    assert y.shape[0] == x.shape[0]
