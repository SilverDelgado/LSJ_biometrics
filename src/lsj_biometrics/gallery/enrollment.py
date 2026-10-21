from dataclasses import dataclass, replace
from typing import Sequence
from uuid import uuid4

from lsj_biometrics.campaign.face import CroppedFaceImage
from lsj_biometrics.campaign.identity import Identity
from lsj_biometrics.campaign.template import Template
from lsj_biometrics.engine.feature_extractor import FeatureExtractor
from lsj_biometrics.gallery.repository import CropRepository, GalleryRepository, require_model


@dataclass(frozen=True)
class EnrollmentResult:
    identity: Identity
    templates: tuple[Template, ...]

class EnrollmentService:
    def __init__(self, extractor: FeatureExtractor, repository: GalleryRepository, crops: CropRepository) -> None:
        self._extractor = extractor
        self._repository = repository
        self._crops = crops

    def enroll(self, identity: Identity, faces: Sequence[CroppedFaceImage]) -> EnrollmentResult:
        templates = self._templates(identity.identity_id, faces)
        self._repository.save_identity(identity)
        self._save(templates)
        return EnrollmentResult(identity=identity, templates=templates)

    def add_faces(self, identity_id: str, faces: Sequence[CroppedFaceImage]) -> tuple[Template, ...]:
        templates = self._templates(identity_id, faces)
        self._save(templates)
        return templates

    def _templates(self, identity_id: str, faces: Sequence[CroppedFaceImage]) -> tuple[Template, ...]:
        if not faces:
            raise ValueError(f"no faces given for identity {identity_id}")
        return tuple(self._template(identity_id, self._crops.save(face)) for face in faces)

    def _template(self, identity_id: str, face: CroppedFaceImage) -> Template:
        template = replace(
            self._extractor.extract(face),
            template_id=str(uuid4()),
            identity_id=identity_id,
            face=face,
        )
        require_model(template, self._repository)
        return template

    def _save(self, templates: tuple[Template, ...]) -> None:
        for template in templates:
            self._repository.save_template(template)