from typing import Protocol

from lsj_biometrics.campaign.face import CroppedFaceImage
from lsj_biometrics.campaign.identity import Identity
from lsj_biometrics.campaign.template import Template
from lsj_biometrics.engine.matcher import Match

class GalleryRepository(Protocol):
    model_key: str

    def save_identity(self, identity: Identity) -> None:
        ...

    def save_template(self, template: Template) -> None:
        ...

    def search(self, template: Template, rank: int = 5) -> list[Match]:
        ...

class CropRepository(Protocol):
    def save(self, face: CroppedFaceImage) -> CroppedFaceImage:
        ...

def require_model(template: Template, repository: GalleryRepository) -> None:
    if template.model_key != repository.model_key:
        raise ValueError(f"template from {template.model_key} cannot be used with a {repository.model_key} gallery")