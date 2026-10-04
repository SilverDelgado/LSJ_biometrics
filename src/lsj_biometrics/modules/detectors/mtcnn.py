from typing import Any, Mapping

import numpy as np
from mtcnn import MTCNN
from PIL import Image as PillowImage

from lsj_biometrics.campaign.face import BoundingBox, Face
from lsj_biometrics.campaign.image import Image
from lsj_biometrics.engine.face_detector import DetectionResult

class MTCNNFaceDetector:
    def __init__(self, detector: MTCNN | None = None) -> None:
        self._detector = detector or MTCNN()

    def detect(self, image: Image) -> DetectionResult:
        with PillowImage.open(image.path) as source:
            rgb_image = np.asarray(source.convert("RGB"))

        detections: list[Mapping[str, Any]] = self._detector.detect_faces(rgb_image)
        faces = tuple(
            self._to_face(image, detection, index)
            for index, detection in enumerate(detections)
        )
        return DetectionResult(image=image, faces=faces)

    @staticmethod
    def _to_face(image: Image,detection: Mapping[str, Any],index: int) -> Face:
        x, y, width, height = detection["box"]
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
        )
