import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from lsj_biometrics.training.casia import CasiaDataset
from lsj_biometrics.training.verification import embed

INPUT_SIZE = 112
EMBEDDING_DIM = 512
PROGRESS_EVERY = 200


def conv3x3(inputs: int, outputs: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(inputs, outputs, 3, stride, 1, bias=False)


def conv1x1(inputs: int, outputs: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(inputs, outputs, 1, stride, bias=False)


class IBasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(self, layers: tuple[int, int, int, int], num_features: int = EMBEDDING_DIM) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(64, layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-05)
        self.dropout = nn.Dropout(p=0.0)
        self.fc = nn.Linear(512 * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = nn.Sequential(conv1x1(self.inplanes, planes, stride), nn.BatchNorm2d(planes, eps=1e-05))
        layers = [IBasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(IBasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return self.features(x)


class ArcFaceTeacher(nn.Module):

    def __init__(self, network: IResNet) -> None:
        super().__init__()
        self.network = network

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2:] != (INPUT_SIZE, INPUT_SIZE):
            raise ValueError(f"the teacher takes {INPUT_SIZE}x{INPUT_SIZE} aligned faces, got {tuple(images.shape)}")
        return self.network((images - 127.5) / 127.5)


def load_teacher(path: Path) -> ArcFaceTeacher:
    network = IResNet((3, 4, 14, 3))
    state = torch.load(path, map_location="cpu", weights_only=True)
    network.load_state_dict(state)
    return ArcFaceTeacher(network).eval()


def precompute(dataset: CasiaDataset, model: nn.Module, out_path: Path, batch_size: int, device: str, workers: int) -> None:
    out_path = Path(out_path)
    if out_path.exists():
        raise ValueError(f"{out_path} exists: delete it to compute it again")
    if dataset.augmenter is not None:
        raise ValueError("the teacher embeds the clean images: the dataset carries an augmenter")
    if not np.array_equal(dataset.indices, np.arange(len(dataset))):
        raise ValueError("the teacher rows follow the CASIA index from 0: the dataset has to cover 0..N-1 in order")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_name(out_path.name + ".partial")
    rows = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=(len(dataset), EMBEDDING_DIM))
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.startswith("cuda"))
    started = time.perf_counter()
    done = 0
    for batch, (images, labels, indices) in enumerate(loader):
        embeddings = embed(model, images, device, True)
        rows[indices.numpy()] = embeddings.cpu().numpy().astype(np.float16)
        done += images.shape[0]
        if (batch + 1) % PROGRESS_EVERY == 0:
            elapsed = time.perf_counter() - started
            print(f"[OK] teacher {done}/{len(dataset)} images, {done / elapsed:.0f} images/s")
    rows.flush()
    del rows
    partial.replace(out_path)
    elapsed = time.perf_counter() - started
    print(f"[OK] teacher embeddings {out_path}: {done} x {EMBEDDING_DIM} float16 in {elapsed:.0f} s")
