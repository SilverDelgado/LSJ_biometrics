import math

import torch
import torch.nn.functional as F
from torch import nn

EPSILON = 1e-3


class AdaFace(nn.Module):

    def __init__(self, embedding_dim: int, num_classes: int, margin: float = 0.4, h: float = 0.333, scale: float = 64.0, t_alpha: float = 0.01) -> None:
        super().__init__()
        self.margin = margin
        self.h = h
        self.scale = scale
        self.t_alpha = t_alpha
        self.kernel = nn.Parameter(torch.empty(embedding_dim, num_classes))
        with torch.no_grad():
            self.kernel.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.register_buffer("batch_mean", torch.ones(1) * 20)
        self.register_buffer("batch_std", torch.ones(1) * 100)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        norms = torch.norm(features, p=2, dim=1, keepdim=True)
        embeddings = features / norms.clamp(min=1e-12)
        cosine = embeddings @ F.normalize(self.kernel, dim=0)
        cosine = cosine.clamp(-1 + EPSILON, 1 - EPSILON)
        safe_norms = torch.clip(norms, min=0.001, max=100).detach()
        with torch.no_grad():
            self.batch_mean.mul_(1 - self.t_alpha).add_(safe_norms.mean() * self.t_alpha)
            self.batch_std.mul_(1 - self.t_alpha).add_(safe_norms.std() * self.t_alpha)
        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + EPSILON)
        margin_scaler = torch.clip(margin_scaler * self.h, -1, 1)
        target = torch.zeros_like(cosine)
        target.scatter_(1, labels.reshape(-1, 1), 1.0)
        theta = cosine.acos()
        theta = torch.clip(theta + target * (-self.margin * margin_scaler), min=EPSILON, max=math.pi - EPSILON)
        cosine = theta.cos() - target * (self.margin + self.margin * margin_scaler)
        return cosine * self.scale
