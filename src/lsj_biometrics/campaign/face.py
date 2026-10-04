from dataclasses import dataclass

@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

@dataclass(frozen=True)
class Face:
    face_id: str
    image_id: str
    bounding_box: BoundingBox
    confidence: float

@dataclass(frozen=True)
class FaceImage:
    face_id: str
    path: str