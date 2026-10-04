from dataclasses import dataclass, replace
from uuid import uuid4

from lsj_biometrics.campaign.face import CroppedFaceImage
from lsj_biometrics.campaign.identity import Identity
from lsj_biometrics.campaign.template import Template
from lsj_biometrics.engine.feature_extractor import FeatureExtractor
from lsj_biometrics.gallery.repository import GalleryRepository


@dataclass(frozen=True)
class EnrollmentResult:
    identity: Identity
    template: Template

class EnrollmentService:
    def __init__(self, extractor: FeatureExtractor, repository: GalleryRepository) -> None:
        self._extractor = extractor
        self._repository = repository

    def enroll(self, identity: Identity, face: CroppedFaceImage) -> EnrollmentResult:
        template = replace(
            self._extractor.extract(face),
            template_id=str(uuid4()),
            identity_id=identity.identity_id,
            face=face,
        )
        self._repository.save_identity(identity)
        self._repository.save_template(template)
        return EnrollmentResult(identity=identity, template=template)