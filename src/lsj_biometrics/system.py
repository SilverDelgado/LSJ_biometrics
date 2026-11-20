import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from lsj_biometrics.campaign.face import CroppedFaceImage, Face
from lsj_biometrics.campaign.identity import Identity
from lsj_biometrics.campaign.image import Image
from lsj_biometrics.campaign.template import Template
from lsj_biometrics.config.operating_point import OperatingPoint
from lsj_biometrics.engine.face_detector import FaceDetector
from lsj_biometrics.engine.feature_extractor import BatchFeatureExtractor
from lsj_biometrics.engine.pipeline import FacePreprocessor
from lsj_biometrics.gallery.enrollment import EnrollmentResult, EnrollmentService
from lsj_biometrics.gallery.identification import BatchIdentification, IdentificationResult, IdentificationService
from lsj_biometrics.gallery.repository import CropRepository, GalleryRepository
from lsj_biometrics.gallery.verification import VerificationResult, VerificationService
from lsj_biometrics.image.loading import load_rgb
from lsj_biometrics.image.preprocess import AlignedPreprocessor
from lsj_biometrics.modules.detectors.mtcnn import MTCNNFaceDetector
from lsj_biometrics.modules.detectors.yunet import MODEL as YUNET_MODEL
from lsj_biometrics.modules.detectors.yunet import YuNetFaceDetector
from lsj_biometrics.modules.extractors.ssm import SSMFeatureExtractor, load_ssm_face_model
from lsj_biometrics.modules.storage.crops import FileCropRepository
from lsj_biometrics.modules.storage.csv_gallery import CsvGalleryRepository


class BiometricSystem:
    def __init__(self, detector: FaceDetector, preprocessor: FacePreprocessor, extractor: BatchFeatureExtractor, repository: GalleryRepository, crops: CropRepository, threshold: float, rank: int, dimension: int) -> None:
        self._detector = detector
        self._preprocessor = preprocessor
        self._extractor = extractor
        self._rank = rank
        self._dimension = dimension
        self._enrollment = EnrollmentService(repository, crops)
        self._verification = VerificationService(repository, threshold)
        self._identification = IdentificationService(repository, threshold)

    def inscribir(self, identity_id: str, foto: Image) -> EnrollmentResult:
        return self._enrollment.enroll(Identity(identity_id), [self._detected(foto)])

    def verificar(self, identity_id: str, foto: Image) -> VerificationResult:
        return self._verification.verify(identity_id, self._detected(foto))

    def identificar(self, foto: Image) -> IdentificationResult:
        return self._identification.identify(self._detected(foto), self._rank)

    def inscribir_sd(self, identity_id: str, foto: Image) -> EnrollmentResult:
        return self._enrollment.enroll(Identity(identity_id), [self._whole(foto)])

    def verificar_sd(self, identity_id: str, foto: Image) -> VerificationResult:
        return self._verification.verify(identity_id, self._whole(foto))

    def identificar_sd(self, foto: Image) -> IdentificationResult:
        return self._identification.identify(self._whole(foto), self._rank)

    def inscribir_t(self, identity_id: str, template: Sequence[float]) -> EnrollmentResult:
        return self._enrollment.enroll(Identity(identity_id), [self._given(template)])

    def verificar_t(self, identity_id: str, template: Sequence[float]) -> VerificationResult:
        return self._verification.verify(identity_id, self._given(template))

    def identificar_t(self, template: Sequence[float]) -> IdentificationResult:
        return self._identification.identify(self._given(template), self._rank)

    def identify_batch(self, vectors: np.ndarray) -> BatchIdentification:
        return self._identification.identify_batch(vectors)

    def _detected(self, foto: Image) -> Template:
        faces = self._detector.detect(foto).faces
        if not faces:
            raise ValueError(f"no face detected in {foto.image_id}")
        return self._extracted(self._preprocessor.crop(largest_face(faces)))

    def _whole(self, foto: Image) -> Template:
        size = self._extractor.layout.size
        pixels = load_rgb(foto, size=(size, size))
        face_id = f"{foto.image_id}:whole"
        face = CroppedFaceImage(face_id=face_id, image=Image(image_id=face_id, source_image_id=foto.image_id, pixels=pixels))
        return self._extracted(face)

    def _extracted(self, face: CroppedFaceImage) -> Template:
        return replace(self._extractor.extract(face), face=face)

    def _given(self, template: Sequence[float]) -> Template:
        values = np.asarray(template, dtype=np.float64)
        if values.ndim != 1 or values.shape[0] != self._dimension:
            raise ValueError(f"a template is {self._dimension} floats, got shape {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("a template must be finite")
        norm = np.linalg.norm(values)
        if norm == 0.0:
            raise ValueError("a zero template has no direction to compare")
        normalized = (values / norm).astype(np.float32)
        return Template(vector=tuple(normalized.tolist()), model_name=self._extractor.model_name, model_version=self._extractor.model_version, normalized=True, created_at=datetime.now(timezone.utc))


def largest_face(faces: Sequence[Face]) -> Face:
    largest = faces[0]
    for face in faces[1:]:
        if face.bounding_box.width * face.bounding_box.height > largest.bounding_box.width * largest.bounding_box.height:
            largest = face
    return largest


def build_detector(name: str) -> FaceDetector:
    if name == "yunet":
        return YuNetFaceDetector(YUNET_MODEL)
    if name == "mtcnn":
        return MTCNNFaceDetector()
    raise ValueError(f"unknown detector {name!r}")


def build_system(point: OperatingPoint) -> BiometricSystem:
    if not point.model.is_file():
        raise FileNotFoundError(f"model checkpoint {point.model} not found, set model in the operating point")
    model = load_ssm_face_model(point.model)
    extractor = SSMFeatureExtractor(model, device=point.device, batch_size=1)
    model_key = f"{extractor.model_name}:{extractor.model_version}"
    repository = CsvGalleryRepository(point.database, model_key)
    detector = build_detector(point.detector)
    preprocessor = AlignedPreprocessor(extractor.layout)
    crops = FileCropRepository(point.crops)
    return BiometricSystem(detector, preprocessor, extractor, repository, crops, point.threshold, point.rank, model.config.embedding_dim)


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

YEAR_SUFFIX = re.compile(r"_\d{4}$")


def identity_from_filename(path: Path) -> str:
    return YEAR_SUFFIX.sub("", path.stem)


def photo(path: Path) -> Image:
    if not path.is_file():
        raise FileNotFoundError(f"photo {path} not found")
    return Image(image_id=path.stem, path=path)


def enroll_folder(system: BiometricSystem, folder: Path) -> tuple[list[EnrollmentResult], list[str]]:
    paths = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            paths.append(path)
        else:
            print(f"[WARN] {path.name}: not a photo, skipped")
    results = []
    failed = []
    for path in paths:
        try:
            results.append(system.inscribir(identity_from_filename(path), photo(path)))
        except ValueError as error:
            print(f"[WARN] {path.name}: {error}")
            failed.append(path.name)
    return results, failed
