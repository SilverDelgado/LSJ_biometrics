from dataclasses import dataclass
from datetime import datetime

import numpy as np

from lsj_biometrics.campaign.image import Image

@dataclass(frozen=True)
class Frame:
    camera_id: str
    index: int
    captured_at: datetime
    stream_seconds: float
    image: Image

    def __post_init__(self) -> None:
        if self.captured_at.tzinfo is None:
            raise ValueError(f"frame {self.camera_id}:{self.index} captured_at must be timezone-aware")
        if self.image.image_id != frame_image_id(self.camera_id, self.index):
            raise ValueError(f"frame image id {self.image.image_id} does not identify camera {self.camera_id} frame {self.index}")

    @classmethod
    def capture(cls, camera_id: str, index: int, captured_at: datetime, stream_seconds: float, pixels: np.ndarray) -> "Frame":
        image = Image(image_id=frame_image_id(camera_id, index), pixels=pixels)
        return cls(camera_id, index, captured_at, stream_seconds, image)


def frame_image_id(camera_id: str, index: int) -> str:
    return f"{camera_id}:{index}"