from dataclasses import fields
from typing import Any, Mapping, Sequence

from lsj_biometrics.campaign.face import BoundingBox, Face, Landmarks
from lsj_biometrics.campaign.image import Image
from lsj_biometrics.engine.face_detector import DetectionResult
from lsj_biometrics.image.loading import load_rgb

class MTCNNFaceDetector:
    def __init__(self, detector: Any | None = None) -> None:
        if detector is None:
            from mtcnn import MTCNN
            detector = MTCNN()
        self._detector = detector

    def detect(self, image: Image) -> DetectionResult:
        detections: list[Mapping[str, Any]] = self._detector.detect_faces(load_rgb(image))
        faces = tuple(
            self._to_face(image, detection, index)
            for index, detection in enumerate(detections)
        )
        return DetectionResult(image=image, faces=faces)

    def detect_batch(self, images: Sequence[Image]) -> list[DetectionResult]:
        return [self.detect(image) for image in images]

    @staticmethod
    def _to_face(image: Image,detection: Mapping[str, Any],index: int) -> Face:
        x, y, width, height = detection["box"]
        keypoints = detection.get("keypoints")
        return Face(
            face_id=f"{image.image_id}:face:{index}",
            image=image,
            bounding_box=BoundingBox(
                x=float(x),
                y=float(y),
                width=float(width),
                height=float(height),
            ),
            confidence=float(detection["confidence"]),
            landmarks=Landmarks(**{
                field.name: (float(keypoints[field.name][0]), float(keypoints[field.name][1]))
                for field in fields(Landmarks)
            }) if keypoints else None,
        )
