from pathlib import Path

import numpy as np
from PIL import Image as PillowImage


def open_rgb(path: Path) -> PillowImage.Image:
    with PillowImage.open(path) as source:
        return source.convert("RGB")


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = open_rgb(path)
    if size is not None:
        image = image.resize(size, PillowImage.Resampling.BILINEAR)
    return np.asarray(image)
