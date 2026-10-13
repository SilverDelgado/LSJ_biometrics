import math

import torch
import torch.nn.functional as F
from torch import nn


class SelectiveSSM(nn.Module):
    def __init__(self, channels: int, state_dim: int = 16, step_range: tuple[float, float] = (1e-3, 1e-1)) -> None:
        super().__init__()
        self.bc_proj = nn.Linear(channels, 2 * state_dim, bias=False)
        self.delta_proj = nn.Linear(channels, channels, bias=False)
        self.a_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32)).repeat(channels, 1))
        self.delta_bias = nn.Parameter(self._initial_delta_bias(channels, *step_range))
        self.skip = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = -torch.exp(self.a_log)
        b, c = self.bc_proj(x).chunk(2, dim=-1)
        delta = F.softplus(self.delta_proj(x) + self.delta_bias)
        decay = torch.exp(delta.unsqueeze(-1) * a)
        update = (delta * x).unsqueeze(-1) * b.unsqueeze(2)

        state = torch.zeros_like(decay[:, 0])
        states = []
        for index in range(x.shape[1]):
            state = decay[:, index] * state + update[:, index]
            states.append(state)

        return torch.einsum("bldn,bln->bld", torch.stack(states, dim=1), c) + x * self.skip

    @staticmethod
    def _initial_delta_bias(channels: int, low: float, high: float) -> torch.Tensor:
        steps = torch.exp(torch.empty(channels).uniform_(math.log(low), math.log(high)))
        return steps + torch.log(-torch.expm1(-steps))


class MambaBlock(nn.Module):
    def __init__(self, dim: int, state_dim: int = 16, expand: int = 2, conv_kernel: int = 4, reverse: bool = False) -> None:
        super().__init__()
        inner = dim * expand
        self.reverse = reverse
        self.in_proj = nn.Linear(dim, 2 * inner, bias=False)
        self.conv = nn.Conv1d(inner, inner, conv_kernel, padding=conv_kernel - 1, groups=inner)
        self.ssm = SelectiveSSM(inner, state_dim)
        self.out_proj = nn.Linear(inner, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reverse:
            x = x.flip(1)
        hidden, gate = self.in_proj(x).chunk(2, dim=-1)
        hidden = self.conv(hidden.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        hidden = self.ssm(F.silu(hidden))
        out = self.out_proj(hidden * F.silu(gate))
        return out.flip(1) if self.reverse else out


class BidirectionalMambaBlock(nn.Module):
    def __init__(self, dim: int, state_dim: int = 16, expand: int = 2, conv_kernel: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.forward_scan = MambaBlock(dim, state_dim, expand, conv_kernel)
        self.backward_scan = MambaBlock(dim, state_dim, expand, conv_kernel, reverse=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        return x + self.forward_scan(normed) + self.backward_scan(normed)
