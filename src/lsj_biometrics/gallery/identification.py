from dataclasses import dataclass

from lsj_biometrics.campaign.face import FaceImage
from lsj_biometrics.campaign.template import Template
from lsj_biometrics.engine.feature_extractor import FeatureExtractor
from lsj_biometrics.engine.matcher import Match
from lsj_biometrics.gallery.repository import GalleryRepository


@dataclass(frozen=True)
class IdentificationResult:
    template: Template
    matches: list[Match]


class IdentificationService:
    def __init__(self, extractor: FeatureExtractor, repository: GalleryRepository) -> None:
        self._extractor = extractor
        self._repository = repository

    def identify(self, face: FaceImage, rank: int = 5) -> IdentificationResult:
        template = self._extractor.extract(face)
        return IdentificationResult(
            template=template,
            matches=self._repository.search(template, rank),
        )