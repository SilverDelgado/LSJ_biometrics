from pathlib import Path

import numpy as np
from PIL import Image as PillowImage
from PIL import ImageOps

from lsj_biometrics.campaign.image import Image

def open_rgb(source: Path | Image) -> PillowImage.Image:
    if isinstance(source, Image):
        if source.pixels is not None:
            return PillowImage.fromarray(source.pixels)
        source = source.path
    with PillowImage.open(source) as image:
        return ImageOps.exif_transpose(image).convert("RGB")

def load_rgb(source: Path | Image, size: tuple[int, int] | None = None) -> np.ndarray:
    if isinstance(source, Image) and source.pixels is not None and size is None:
        return source.pixels
    image = open_rgb(source)
    if size is not None and image.size != size:
        image = image.resize(size, PillowImage.Resampling.BILINEAR)
    return np.asarray(image)
