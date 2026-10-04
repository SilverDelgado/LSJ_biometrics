from typing import Protocol

from lsj_biometrics.campaign.face import CroppedFaceImage, Face

class FacePreprocessor(Protocol):
    def crop(self, face: Face) -> CroppedFaceImage:
        ...