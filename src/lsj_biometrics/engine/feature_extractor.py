from typing import Protocol

from lsj_biometrics.campaign.face import FaceImage
from lsj_biometrics.campaign.template import Template

class FeatureExtractor(Protocol):
    def extract(self, face: FaceImage) -> Template:
        ...