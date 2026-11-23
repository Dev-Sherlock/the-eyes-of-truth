from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="EOT_", extra="ignore")

    seed: int = 42
    device: str = "cpu"
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"
    mlflow_tracking_uri: str = "sqlite:///artifacts/mlflow.db"

    # data
    batch_size: int = 32
    num_workers: int = 0
    image_size: int = 32

    # dino/pretrain (CPU-friendly)
    backbone: str = "resnet18"
    out_dim: int = 256
    dino_epochs: int = 3
    dino_lr: float = 5e-4
    dino_weight_decay: float = 0.04
    teacher_momentum: float = 0.996
    n_global_crops: int = 2
    n_local_crops: int = 2
    global_crop_size: int = 24
    local_crop_size: int = 16

    # classifier
    num_classes: int = 10
    finetune_epochs: int = 3
    finetune_lr: float = 1e-3

    # api/app
    model_path: str = "artifacts/classifier.pt"
    backbone_path: str = "artifacts/dino_backbone.pt"


def get_settings() -> Settings:
    return Settings()
