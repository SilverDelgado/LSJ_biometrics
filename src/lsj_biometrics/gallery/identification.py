from dataclasses import dataclass
from typing import Iterable

from lsj_biometrics.campaign.face import CroppedFaceImage
from lsj_biometrics.campaign.template import Template
from lsj_biometrics.engine.feature_extractor import FeatureExtractor
from lsj_biometrics.engine.matcher import Match
from lsj_biometrics.gallery.repository import GalleryRepository, require_model


@dataclass(frozen=True)
class IdentificationResult:
    template: Template
    matches: list[Match]
    decision: Match | None

class IdentificationService:
    def __init__(self, extractor: FeatureExtractor, repository: GalleryRepository, threshold: float) -> None:
        self._extractor = extractor
        self._repository = repository
        self._threshold = threshold

    def identify(self, face: CroppedFaceImage, rank: int = 5) -> IdentificationResult:
        template = self._extractor.extract(face)
        require_model(template, self._repository)
        matches = best_per_identity(self._repository.search(template, rank))[:rank]
        decision = matches[0] if matches and matches[0].score >= self._threshold else None
        return IdentificationResult(template=template, matches=matches, decision=decision)


def best_per_identity(matches: Iterable[Match]) -> list[Match]:
    best: dict[str, Match] = {}
    for match in matches:
        if match.identity_id not in best or match.score > best[match.identity_id].score:
            best[match.identity_id] = match
    return sorted(best.values(), key=lambda match: match.score, reverse=True)