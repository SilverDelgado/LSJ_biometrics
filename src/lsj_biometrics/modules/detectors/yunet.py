from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from lsj_biometrics.campaign.face import BoundingBox, Face, Landmarks
from lsj_biometrics.campaign.image import Image
from lsj_biometrics.engine.face_detector import DetectionResult
from lsj_biometrics.image.loading import load_rgb

MODEL = Path(__file__).resolve().parents[4] / "models" / "yunet" / "face_detection_yunet_2023mar.onnx"

MAX_SIDE = 640

SCORE_THRESHOLD = 0.8
NMS_THRESHOLD = 0.3
TOP_K = 5000


class YuNetFaceDetector:
    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"yunet model not found at {model_path}")
        self._detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K)

    def detect(self, image: Image) -> DetectionResult:
        pixels = load_rgb(image)
        height, width = pixels.shape[:2]
        scale = min(1.0, MAX_SIDE / max(height, width))
        bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        if scale < 1.0:
            bgr = cv2.resize(bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        self._detector.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, rows = self._detector.detect(bgr)
        faces = []
        if rows is not None:
            for index, row in enumerate(rows):
                faces.append(yunet_face(image, row, scale, index))
        return DetectionResult(image=image, faces=tuple(faces))

    def detect_batch(self, images: Sequence[Image]) -> list[DetectionResult]:
        results = []
        for image in images:
            results.append(self.detect(image))
        return results


def yunet_face(image: Image, row: np.ndarray, scale: float, index: int) -> Face:
    values = []
    for value in row[:14]:
        values.append(float(value) / scale)
    box = BoundingBox(x=values[0], y=values[1], width=values[2], height=values[3])
    landmarks = Landmarks(left_eye=(values[4], values[5]), right_eye=(values[6], values[7]), nose=(values[8], values[9]), mouth_left=(values[10], values[11]), mouth_right=(values[12], values[13]))
    return Face(face_id=f"{image.image_id}:face:{index}", image=image, bounding_box=box, confidence=float(row[14]), landmarks=landmarks)
