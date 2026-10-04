from dataclasses import dataclass
from typing import Protocol

from lsj_biometrics.campaign.face import Face
from lsj_biometrics.campaign.image import Image

@dataclass(frozen=True)
class DetectionResult:
    image: Image
    faces: tuple[Face, ...] #tupla porque no se modificará lo que aparece

class FaceDetector(Protocol):
    def detect(self, image: Image) -> DetectionResult:
        ...