from dataclasses import dataclass

from lsj_biometrics.campaign.template import Template
from lsj_biometrics.gallery.decision import accepts
from lsj_biometrics.gallery.repository import GalleryRepository, probe_vector, require_model


@dataclass(frozen=True)
class VerificationResult:
    identity_id: str
    score: float
    accepted: bool


class VerificationService:
    def __init__(self, repository: GalleryRepository, threshold: float) -> None:
        self._repository = repository
        self._threshold = threshold

    def verify(self, identity_id: str, template: Template) -> VerificationResult:
        require_model(template, self._repository)
        scores, enrolled = self._repository.scores(probe_vector(template))
        if identity_id not in enrolled:
            raise ValueError(f"identity {identity_id} is not enrolled")
        score = float(scores[0, enrolled.index(identity_id)])
        return VerificationResult(identity_id=identity_id, score=score, accepted=bool(accepts(score, self._threshold)))
