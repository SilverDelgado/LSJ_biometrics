import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from lsj_biometrics.training.augment import Perturber
from lsj_biometrics.training.casia import IMAGE_SIZE, decode_rgb

FOLDS = 10
FAR = 1e-3
THRESHOLDS = np.arange(0, 4, 0.01)


@dataclass(frozen=True)
class PairSet:
    name: str
    images: np.ndarray
    issame: np.ndarray
    pairs: np.ndarray


@dataclass(frozen=True)
class VerificationResult:
    name: str
    condition: str
    pairs: int
    accuracy: float
    accuracy_std: float
    threshold: float
    tar_at_far: float

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "condition": self.condition,
            "pairs": self.pairs,
            "accuracy": self.accuracy,
            "accuracy_std": self.accuracy_std,
            "threshold": self.threshold,
            "tar_at_far": self.tar_at_far,
        }


def read_pairs(path: Path, limit: int | None) -> PairSet:
    path = Path(path)
    with open(path, "rb") as handle:
        bins, issame = pickle.load(handle, encoding="bytes")
    total = len(issame)
    if len(bins) != 2 * total:
        raise ValueError(f"{path} has {len(bins)} images for {total} pairs")
    if limit is None:
        chosen = np.arange(total)
    else:
        if limit < 2 * FOLDS or limit > total:
            raise ValueError(f"limit {limit} outside {2 * FOLDS}..{total} pairs")
        chosen = (np.arange(limit) * total) // limit
    images = np.empty((2 * len(chosen), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    same = np.zeros(len(chosen), dtype=bool)
    for position, pair in enumerate(chosen):
        same[position] = bool(issame[pair])
        for side in (0, 1):
            image = decode_rgb(bins[2 * pair + side])
            if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
                raise ValueError(f"{path} pair {pair} has an image of {image.shape}")
            images[2 * position + side] = image
    return PairSet(name=path.stem, images=images, issame=same, pairs=chosen)


def perturbed(pairs: PairSet, perturber: Perturber, transform: str, level: int) -> PairSet:
    images = pairs.images.copy()
    for position, pair in enumerate(pairs.pairs):
        images[2 * position + 1] = perturber.apply(images[2 * position + 1], transform, level, int(pair))
    return PairSet(name=pairs.name, images=images, issame=pairs.issame, pairs=pairs.pairs)


def embed(model: nn.Module, images: torch.Tensor, device: str, flip: bool) -> torch.Tensor:
    x = images.to(device, non_blocking=True).permute(0, 3, 1, 2).float()
    with torch.no_grad():
        features = model(x).float()
        if flip:
            features = features + model(torch.flip(x, dims=[3])).float()
    return F.normalize(features, dim=1)


def similarities(model: nn.Module, pairs: PairSet, batch_size: int, device: str, flip: bool) -> np.ndarray:
    chunks = []
    for start in range(0, pairs.images.shape[0], batch_size):
        batch = torch.from_numpy(pairs.images[start:start + batch_size])
        chunks.append(embed(model, batch, device, flip).cpu().numpy())
    embeddings = np.concatenate(chunks, axis=0)
    return np.sum(embeddings[0::2] * embeddings[1::2], axis=1)


def ten_fold_accuracy(similarity: np.ndarray, issame: np.ndarray) -> tuple[float, float, float]:
    distances = 2.0 - 2.0 * similarity
    folds = np.array_split(np.arange(len(issame)), FOLDS)
    accuracies = []
    thresholds = []
    for test in folds:
        train = np.ones(len(issame), dtype=bool)
        train[test] = False
        predicted = distances[train][:, None] < THRESHOLDS[None, :]
        train_accuracy = np.mean(predicted == issame[train][:, None], axis=0)
        best = THRESHOLDS[int(np.argmax(train_accuracy))]
        accuracies.append(float(np.mean((distances[test] < best) == issame[test])))
        thresholds.append(1.0 - best / 2.0)
    return float(np.mean(accuracies)), float(np.std(accuracies)), float(np.mean(thresholds))


def tar_at_far(similarity: np.ndarray, issame: np.ndarray, far: float) -> float:
    negatives = np.sort(similarity[~issame])[::-1]
    positives = similarity[issame]
    if negatives.size == 0 or positives.size == 0:
        raise ValueError("TAR@FAR needs positive and negative pairs")
    allowed = int(far * negatives.size)
    threshold = negatives[allowed]
    return float(np.mean(positives > threshold))


def verify(model: nn.Module, pairs: PairSet, condition: str, batch_size: int, device: str, flip: bool) -> VerificationResult:
    similarity = similarities(model, pairs, batch_size, device, flip)
    accuracy, accuracy_std, threshold = ten_fold_accuracy(similarity, pairs.issame)
    tar = tar_at_far(similarity, pairs.issame, FAR)
    return VerificationResult(name=pairs.name, condition=condition, pairs=len(pairs.issame), accuracy=accuracy, accuracy_std=accuracy_std, threshold=threshold, tar_at_far=tar)


def evaluate_sets(model: nn.Module, eval_dir: Path, names: tuple[str, ...], limit: int | None, batch_size: int, device: str, flip: bool, perturber: Perturber | None) -> list[VerificationResult]:
    was_training = model.training
    model.eval()
    results = []
    for name in names:
        pairs = read_pairs(Path(eval_dir) / f"{name}.bin", limit)
        results.append(verify(model, pairs, "clean", batch_size, device, flip))
        if perturber is not None:
            for transform in sorted(perturber.levels):
                for level in perturber.levels[transform]:
                    condition = f"{transform}_{level}"
                    results.append(verify(model, perturbed(pairs, perturber, transform, level), condition, batch_size, device, flip))
    model.train(was_training)
    return results


def print_table(results: list[VerificationResult]) -> None:
    print(f"{'set':<10} {'condition':<24} {'pairs':>6} {'accuracy':>9} {'std':>7} {'TAR@FAR=1e-3':>13}")
    for result in results:
        print(f"{result.name:<10} {result.condition:<24} {result.pairs:>6} {result.accuracy:>9.4f} {result.accuracy_std:>7.4f} {result.tar_at_far:>13.4f}")

