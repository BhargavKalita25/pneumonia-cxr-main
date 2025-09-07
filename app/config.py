import os, yaml
from dataclasses import dataclass, field

@dataclass
class TrainConfig:
    data_root: str = "data/kaggle_chest_xray"
    output_dir: str = "outputs"
    model_name: str = "densenet121"  # DEFAULT best model
    img_size: int = 224
    batch_size: int = 64
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    loss: str = "bce"  # "bce" or "focal"
    class_weight: float = 0.0
    amp: bool = True
    seed: int = 42
    num_workers: int = 4
    freeze_backbone_epochs: int = 1
    early_stop_patience: int = 4
    profile: str = "gpu_t4"

@dataclass
class ServeConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_mb: int = 10
    allowed_ext: tuple = (".png", ".jpg", ".jpeg", ".dcm")
    model_registry: dict = field(default_factory=lambda: {
        "default": {"path": "outputs/best.ckpt", "arch": "densenet121", "img_size": 224}
    })

def load_profile(cfg: TrainConfig) -> TrainConfig:
    path = os.path.join(os.path.dirname(__file__), "profiles", f"{cfg.profile}.yaml")
    if os.path.exists(path):
        with open(path) as f:
            prof = yaml.safe_load(f)
        for k,v in prof.items():
            setattr(cfg, k, v)
    return cfg
