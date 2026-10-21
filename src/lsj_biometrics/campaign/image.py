from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

@dataclass(frozen=True)
class Image:
    image_id: str
    path: Optional[Path] = None
    sha256: Optional[str] = None
    source_image_id: Optional[str] = None #si la imagen es recortada y viene de esta source
    pixels: Optional[np.ndarray] = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.pixels is None:
            if self.path is None:
                raise ValueError(f"image {self.image_id} has neither path nor pixels")
            return
        if self.pixels.dtype != np.uint8 or self.pixels.ndim != 3 or self.pixels.shape[2] != 3:
            raise ValueError(
                f"image {self.image_id} pixels must be uint8 RGB with shape (height, width, 3), "
                f"got {self.pixels.dtype} {self.pixels.shape}"
            )
        self.pixels.flags.writeable = False