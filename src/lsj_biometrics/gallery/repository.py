from typing import Protocol, Sequence

import numpy as np

from lsj_biometrics.campaign.face import CroppedFaceImage
from lsj_biometrics.campaign.template import Template
from lsj_biometrics.engine.matcher import Match


class GalleryRepository(Protocol):
    model_key: str

    def save_templates(self, templates: Sequence[Template]) -> None:
        ...

    def scores(self, vectors: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
        ...

    def search(self, template: Template, rank: int) -> list[Match]:
        ...


class CropRepository(Protocol):
    def save(self, face: CroppedFaceImage) -> CroppedFaceImage:
        ...


def require_model(template: Template, repository: GalleryRepository) -> None:
    if template.model_key != repository.model_key:
        raise ValueError(f"template from {template.model_key} cannot be used with a {repository.model_key} gallery")


def probe_vector(template: Template) -> np.ndarray:
    return np.asarray([template.vector], dtype=np.float32)
