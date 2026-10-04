from typing import Protocol

from lsj_biometrics.campaign.face import Face, FaceImage

class FacePreprocessor(Protocol):
    def crop(self, face: Face) -> FaceImage:
        ...