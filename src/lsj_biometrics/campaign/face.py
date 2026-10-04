from dataclasses import dataclass
from typing import Optional

from lsj_biometrics.campaign.image import Image

@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

@dataclass(frozen=True)
class Face:
    face_id: str
    image: Image
    bounding_box: BoundingBox
    confidence: Optional[float] = None

@dataclass(frozen=True)
class CroppedFaceImage:
    face_id: str
    image: Image