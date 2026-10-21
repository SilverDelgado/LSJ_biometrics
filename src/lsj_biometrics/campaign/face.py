import math
from dataclasses import dataclass
from typing import Optional

from lsj_biometrics.campaign.image import Image

Point = tuple[float, float]

@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

@dataclass(frozen=True)
class Landmarks:
    left_eye: Point
    right_eye: Point
    nose: Point
    mouth_left: Point
    mouth_right: Point

    def points(self) -> tuple[Point, ...]:
        return (self.left_eye, self.right_eye, self.nose, self.mouth_left, self.mouth_right)

    @property
    def inter_eye_distance(self) -> float:
        return math.dist(self.left_eye, self.right_eye)

@dataclass(frozen=True)
class FaceLayout:
    size: int
    landmarks: Landmarks

@dataclass(frozen=True)
class Face:
    face_id: str
    image: Image
    bounding_box: BoundingBox
    confidence: Optional[float] = None
    landmarks: Optional[Landmarks] = None

@dataclass(frozen=True)
class CroppedFaceImage:
    face_id: str
    image: Image
    layout: Optional[FaceLayout] = None