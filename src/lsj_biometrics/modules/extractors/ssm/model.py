from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from lsj_biometrics.modules.extractors.ssm.layers import VSSDBackbone

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SSMConfig:
    name: str = "ssm-face"
    version: str = "0.1.0"
    image_size: int = 112
    reference_landmarks: tuple[tuple[float, float], ...] = (
        (38.2946, 51.6963),
        (73.5318, 51.5014),
        (56.0252, 71.7366),
        (41.5493, 92.3655),
        (70.7299, 92.2041),
    )
    input_size: int = 224
    depths: tuple[int, ...] = (2, 4, 8, 4)
    dims: tuple[int, ...] = (64, 128, 256, 512)
    d_state: int = 64
    expand: int = 2
    headdim: int = 64
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.2
    dropout: float = 0.0
    embedding_dim: int = 512

    def __post_init__(self) -> None:
        if len(self.depths) != len(self.dims):
            raise ValueError(f"depths {self.depths} and dims {self.dims} must have one entry per stage")
        if self.input_size % self.stride() != 0:
            raise ValueError(f"input_size {self.input_size} must be a multiple of {self.stride()}, the stride of the backbone")
        for dim in self.dims:
            if (dim * self.expand) % self.headdim != 0:
                raise ValueError(f"dim {dim} times expand {self.expand} is not a multiple of headdim {self.headdim}")

    def stride(self) -> int:
        return 4 * 2 ** (len(self.dims) - 1)

    def final_size(self) -> int:
        return self.input_size // self.stride()


class SSMFaceModel(nn.Module):
    def __init__(self, config: SSMConfig = SSMConfig()) -> None:
        super().__init__()
        self.config = config
        self.backbone = VSSDBackbone(config.depths, config.dims, config.expand, config.headdim, config.d_state, config.mlp_ratio, config.drop_path_rate)
        final_dim = config.dims[-1]
        output_layer = [
            nn.BatchNorm2d(final_dim),
            nn.Dropout(config.dropout),
            nn.Flatten(),
            nn.Linear(final_dim * config.final_size() ** 2, config.embedding_dim),
            nn.BatchNorm1d(config.embedding_dim),
        ]
        self.output_layer = nn.Sequential(*output_layer)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1) * 255.0, persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) * 255.0, persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        size = self.config.input_size
        x = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.output_layer(self.backbone(x))

    def set_checkpointing(self, enabled: bool) -> None:
        for stage in self.backbone.layers:
            stage.checkpointing = enabled


def save_ssm_face_model(model: SSMFaceModel, path: Path) -> None:
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict()}, path)


def load_ssm_face_model(path: Path) -> SSMFaceModel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    fields = {}
    for key, value in checkpoint["config"].items():
        fields[key] = as_tuples(value)
    model = SSMFaceModel(SSMConfig(**fields))
    model.load_state_dict(checkpoint["state_dict"])
    return model


def as_tuples(value):
    if not isinstance(value, (list, tuple)):
        return value
    items = []
    for item in value:
        items.append(as_tuples(item))
    return tuple(items)


def load_imagenet_backbone(model: SSMFaceModel, path: Path) -> None:
    with torch.serialization.safe_globals([(OrderedDict, "yacs.config.CfgNode")]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    source = checkpoint["model_ema"]
    target = model.backbone.state_dict()
    last_mixer = f"layers.{len(model.config.depths) - 1}.blocks."
    loaded = {}
    unused = []
    errors = []
    for key, value in source.items():
        if key.startswith("head."):
            unused.append(key)
        elif key not in target:
            if key.startswith(last_mixer) and ".attn." in key:
                unused.append(key)
            else:
                errors.append(f"unexpected {key}")
        elif target[key].shape != value.shape:
            errors.append(f"shape of {key}: checkpoint {tuple(value.shape)}, model {tuple(target[key].shape)}")
        else:
            loaded[key] = value
    fresh = []
    for key in target:
        if key in loaded:
            continue
        if key.startswith(last_mixer) and ".attn." in key:
            fresh.append(key)
        else:
            errors.append(f"missing {key}")
    if errors:
        raise ValueError(f"{path} does not fit the backbone: " + "; ".join(errors))
    model.backbone.load_state_dict(loaded, strict=False)
    print(f"[OK] {path}: {len(loaded)} tensors loaded from model_ema; unused {len(unused)} (stage {len(model.config.depths)} attention and classifier head); random {len(fresh)} (stage {len(model.config.depths)} NC-SSD mixer)")
