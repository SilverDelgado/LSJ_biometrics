from dataclasses import dataclass

import numpy as np

from lsj_biometrics.campaign.template import Template
from lsj_biometrics.engine.matcher import Match
from lsj_biometrics.gallery.decision import accepts
from lsj_biometrics.gallery.repository import GalleryRepository, require_model


@dataclass(frozen=True)
class IdentificationResult:
    template: Template
    matches: list[Match]
    decision: Match | None


@dataclass(frozen=True)
class BatchIdentification:
    enrolled: tuple[str, ...]
    identities: np.ndarray
    scores: np.ndarray


class IdentificationService:
    def __init__(self, repository: GalleryRepository, threshold: float) -> None:
        self._repository = repository
        self._threshold = threshold

    def identify(self, template: Template, rank: int) -> IdentificationResult:
        require_model(template, self._repository)
        matches = self._repository.search(template, rank)
        decision = None
        if matches and accepts(matches[0].score, self._threshold):
            decision = matches[0]
        return IdentificationResult(template=template, matches=matches, decision=decision)

    def identify_batch(self, vectors: np.ndarray) -> BatchIdentification:
        scores, enrolled = self._repository.scores(vectors)
        if not enrolled:
            raise ValueError("the gallery is empty: nothing to identify against")
        best = scores.argmax(axis=1)
        identities = np.asarray(enrolled)[best]
        top = scores[np.arange(scores.shape[0]), best].astype(np.float64)
        return BatchIdentification(enrolled=enrolled, identities=identities, scores=top)
