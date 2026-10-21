from typing import Protocol, Sequence

from lsj_biometrics.campaign.face import CroppedFaceImage, FaceLayout
from lsj_biometrics.campaign.template import Template

class FeatureExtractor(Protocol):
    def extract(self, face: CroppedFaceImage) -> Template:
        ...

class BatchFeatureExtractor(FeatureExtractor, Protocol):
    model_name: str
    model_version: str
    layout: FaceLayout

    def extract_batch(self, faces: Sequence[CroppedFaceImage]) -> list[Template]:
        ...