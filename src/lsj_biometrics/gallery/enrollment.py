from dataclasses import dataclass
from uuid import uuid4

from lsj_biometrics.campaign.face import FaceImage
from lsj_biometrics.campaign.identity import Identity
from lsj_biometrics.campaign.image import Image
from lsj_biometrics.campaign.template import GalleryTemplate
from lsj_biometrics.engine.feature_extractor import FeatureExtractor
from lsj_biometrics.gallery.repository import GalleryRepository


@dataclass(frozen=True)
class EnrollmentResult:
    identity: Identity
    template: GalleryTemplate


class EnrollmentService:
    def __init__(self, extractor: FeatureExtractor, repository: GalleryRepository) -> None:
        self._extractor = extractor
        self._repository = repository

    def enroll(self, image: Image, identity: Identity, face: FaceImage) -> EnrollmentResult:
        template = GalleryTemplate(
            template_id=str(uuid4()),
            identity_id=identity.identity_id,
            image_id=image.image_id,
            template=self._extractor.extract(face),
        )
        self._repository.save_identity(identity)
        self._repository.save_template(template)
        return EnrollmentResult(identity=identity, template=template)