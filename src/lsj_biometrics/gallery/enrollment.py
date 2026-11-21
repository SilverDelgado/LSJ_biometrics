from dataclasses import dataclass, replace
from typing import Sequence
from uuid import uuid4

from lsj_biometrics.campaign.identity import Identity
from lsj_biometrics.campaign.template import Template
from lsj_biometrics.gallery.repository import CropRepository, GalleryRepository, require_model


@dataclass(frozen=True)
class EnrollmentResult:
    identity: Identity
    templates: tuple[Template, ...]


class EnrollmentService:
    def __init__(self, repository: GalleryRepository, crops: CropRepository) -> None:
        self._repository = repository
        self._crops = crops

    def enroll(self, identity: Identity, templates: Sequence[Template]) -> EnrollmentResult:
        if not templates:
            raise ValueError(f"no templates given for identity {identity.identity_id}")
        for template in templates:
            require_model(template, self._repository)
        stored = []
        for template in templates:
            face = template.face
            if face is not None:
                face = self._crops.save(face)
            stored.append(replace(template, template_id=str(uuid4()), identity_id=identity.identity_id, face=face))
        self._repository.save_templates(stored)
        return EnrollmentResult(identity=identity, templates=tuple(stored))
