import math
from dataclasses import dataclass
from pathlib import Path

import yaml

KEYS = (
    "model",
    "detector",
    "device",
    "database",
    "crops",
    "threshold",
    "threshold_origin",
    "rank",
)

DETECTORS = ("yunet", "mtcnn")

PACKAGED = Path(__file__).resolve().parent / "operating_point.yaml"


@dataclass(frozen=True)
class OperatingPoint:
    model: Path
    detector: str
    device: str
    database: Path
    crops: Path
    threshold: float
    threshold_origin: str
    rank: int


def load_operating_point(path: Path) -> OperatingPoint:
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: the operating point must be a mapping")
    missing = []
    for key in KEYS:
        if key not in raw:
            missing.append(key)
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    unknown = []
    for key in raw:
        if key not in KEYS:
            unknown.append(key)
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}")
    if raw["detector"] not in DETECTORS:
        raise ValueError(f"{path}: detector must be one of {DETECTORS}, got {raw['detector']!r}")
    for key in ("model", "device", "database", "crops", "threshold_origin"):
        if not isinstance(raw[key], str) or not raw[key]:
            raise ValueError(f"{path}: {key} must be a non-empty string")
    if isinstance(raw["threshold"], bool) or not isinstance(raw["threshold"], (int, float)):
        raise ValueError(f"{path}: threshold must be a number")
    if not math.isfinite(raw["threshold"]) or raw["threshold"] < -1.0 or raw["threshold"] > 1.0:
        raise ValueError(f"{path}: threshold is a cosine, it must lie in [-1, 1], got {raw['threshold']}")
    if isinstance(raw["rank"], bool) or not isinstance(raw["rank"], int) or raw["rank"] < 1:
        raise ValueError(f"{path}: rank must be a positive integer, got {raw['rank']!r}")
    return OperatingPoint(model=Path(raw["model"]), detector=raw["detector"], device=raw["device"], database=Path(raw["database"]), crops=Path(raw["crops"]), threshold=float(raw["threshold"]), threshold_origin=raw["threshold_origin"], rank=raw["rank"])
