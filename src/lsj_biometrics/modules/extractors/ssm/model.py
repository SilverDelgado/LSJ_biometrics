from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from lsj_biometrics.modules.extractors.ssm.layers import BidirectionalMambaBlock


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
    patch_size: int = 8
    dim: int = 192
    depth: int = 6
    state_dim: int = 16
    expand: int = 2
    conv_kernel: int = 4
    embedding_dim: int = 512


class SSMFaceModel(nn.Module):
    def __init__(self, config: SSMConfig = SSMConfig()) -> None:
        super().__init__()
        self.config = config
        tokens = (config.image_size // config.patch_size) ** 2
        self.patch_embedding = nn.Conv2d(3, config.dim, config.patch_size, stride=config.patch_size)
        self.positions = nn.Parameter(torch.empty(1, tokens, config.dim))
        nn.init.trunc_normal_(self.positions, std=0.02)
        self.blocks = nn.Sequential(*[
            BidirectionalMambaBlock(config.dim, config.state_dim, config.expand, config.conv_kernel)
            for _ in range(config.depth)
        ])
        self.norm = nn.LayerNorm(config.dim)
        self.embedding = nn.Linear(config.dim, config.embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(images / 127.5 - 1.0).flatten(2).transpose(1, 2)
        x = self.blocks(x + self.positions)
        return self.embedding(self.norm(x).mean(dim=1))


def save_ssm_face_model(model: SSMFaceModel, path: Path) -> None:
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict()}, path)


def load_ssm_face_model(path: Path) -> SSMFaceModel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = SSMFaceModel(SSMConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model
