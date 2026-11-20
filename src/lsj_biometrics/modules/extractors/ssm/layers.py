import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


def halved(size: int) -> int:
    return (size - 1) // 2 + 1


class DropPath(nn.Module):
    def __init__(self, drop_prob: float) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask.div_(keep_prob)


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, groups: int = 1, bias: bool = True, norm: bool = True, act: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.norm = nn.BatchNorm2d(out_channels) if norm else None
        self.act = nn.ReLU() if act else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class Stem(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        half = embed_dim // 2
        self.conv1 = ConvLayer(in_channels, half, 3, stride=2, padding=1, bias=False)
        conv2 = [
            ConvLayer(half, half, 3, padding=1, bias=False),
            ConvLayer(half, half, 3, padding=1, bias=False, act=False),
        ]
        self.conv2 = nn.Sequential(*conv2)
        conv3 = [
            ConvLayer(half, embed_dim * 4, 3, stride=2, padding=1, bias=False),
            ConvLayer(embed_dim * 4, embed_dim, 1, bias=False, act=False),
        ]
        self.conv3 = nn.Sequential(*conv3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv3(x)
        return x.flatten(2).transpose(1, 2)


class PatchMerging(nn.Module):
    def __init__(self, dim: int, out_dim: int, ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(out_dim * ratio)
        conv = [
            ConvLayer(dim, hidden, 1, norm=False),
            ConvLayer(hidden, hidden, 3, stride=2, padding=1, groups=hidden, norm=False),
            ConvLayer(hidden, out_dim, 1, act=False),
        ]
        self.conv = nn.Sequential(*conv)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, _, channels = x.shape
        x = x.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        return self.conv(x).flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class NonCausalSSD(nn.Module):
    def __init__(self, d_model: int, expand: int, headdim: int, d_state: int, d_conv: int = 3, a_init_range: tuple[float, float] = (1.0, 16.0), dt_min: float = 0.001, dt_max: float = 0.1, dt_init_floor: float = 1e-4) -> None:
        super().__init__()
        self.d_inner = expand * d_model
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner + 2 * d_state + self.nheads, bias=False)
        conv_dim = self.d_inner + 2 * d_state
        self.conv2d = nn.Conv2d(conv_dim, conv_dim, d_conv, padding=(d_conv - 1) // 2, groups=conv_dim)
        dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.nheads).uniform_(*a_init_range)))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, length, _ = u.shape
        a = -torch.exp(self.A_log)
        z, xbc, dt = torch.split(self.in_proj(u), [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads], dim=-1)
        dt = F.softplus(dt + self.dt_bias)
        xbc = xbc.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
        xbc = F.silu(self.conv2d(xbc))
        xbc = xbc.permute(0, 2, 3, 1).reshape(batch, length, -1).contiguous()
        x, b, c = torch.split(xbc, [self.d_inner, self.d_state, self.d_state], dim=-1)
        y = self.attend(x, dt, a, b, c)
        return self.out_proj(self.norm(y) * z)

    def attend(self, x: torch.Tensor, dt: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        values = x.reshape(batch, length, self.nheads, self.headdim).permute(0, 2, 1, 3)
        decay = -(dt.permute(0, 2, 1).unsqueeze(-1) * a.view(1, -1, 1, 1))
        keys = b.reshape(batch, 1, length, self.d_state)
        queries = c.reshape(batch, 1, length, self.d_state)
        state = keys.transpose(-2, -1) @ (values * decay)
        y = queries @ state + values * self.D.view(1, -1, 1, 1)
        return y.permute(0, 2, 1, 3).reshape(batch, length, self.d_inner)


class VSSDBlock(nn.Module):
    def __init__(self, dim: int, expand: int, headdim: int, d_state: int, mlp_ratio: float, drop_path: float) -> None:
        super().__init__()
        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = NonCausalSSD(dim, expand, headdim, d_state)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = x + self.position(self.cpe1, x, height, width)
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        x = x + self.position(self.cpe2, x, height, width)
        return x + self.drop_path(self.mlp(self.norm2(x)))

    @staticmethod
    def position(conv: nn.Conv2d, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, _, channels = x.shape
        return conv(x.reshape(batch, height, width, channels).permute(0, 3, 1, 2)).flatten(2).transpose(1, 2)


class VSSDStage(nn.Module):
    def __init__(self, dim: int, out_dim: int | None, expand: int, headdim: int, d_state: int, mlp_ratio: float, drop_paths: list[float]) -> None:
        super().__init__()
        blocks = []
        for drop_path in drop_paths:
            blocks.append(VSSDBlock(dim, expand, headdim, d_state, mlp_ratio, drop_path))
        self.blocks = nn.ModuleList(blocks)
        self.downsample = PatchMerging(dim, out_dim) if out_dim is not None else None
        self.checkpointing = False

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if self.checkpointing and self.training:
            x = checkpoint(self.run_blocks, x, height, width, use_reentrant=False)
        else:
            x = self.run_blocks(x, height, width)
        if self.downsample is None:
            return x
        return self.downsample(x, height, width)

    def run_blocks(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, height, width)
        return x


class VSSDBackbone(nn.Module):
    def __init__(self, depths: tuple[int, ...], dims: tuple[int, ...], expand: int, headdim: int, d_state: int, mlp_ratio: float, drop_path_rate: float) -> None:
        super().__init__()
        self.patch_embed = Stem(3, dims[0])
        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        layers = []
        start = 0
        for index in range(len(depths)):
            out_dim = dims[index + 1] if index + 1 < len(dims) else None
            layers.append(VSSDStage(dims[index], out_dim, expand, headdim, d_state, mlp_ratio, rates[start : start + depths[index]]))
            start = start + depths[index]
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(dims[-1])
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        height = halved(halved(images.shape[2]))
        width = halved(halved(images.shape[3]))
        x = self.patch_embed(images)
        for stage in self.layers:
            x = stage(x, height, width)
            if stage.downsample is not None:
                height = halved(height)
                width = halved(width)
        x = self.norm(x)
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], height, width)
