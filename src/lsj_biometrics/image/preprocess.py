import hashlib
import re
from pathlib import Path

import numpy as np
from PIL import Image as PillowImage

from lsj_biometrics.campaign.face import BoundingBox, CroppedFaceImage, Face, FaceLayout
from lsj_biometrics.campaign.image import Image
from lsj_biometrics.engine.pipeline import FacePreprocessor
from lsj_biometrics.image.alignment import align_face
from lsj_biometrics.image.loading import load_rgb, open_rgb


class ImagePreprocessor:
    def __init__(self, size: int, output_dir: Path | None = None, margin: float = 0.2) -> None:
        self._output_dir = output_dir
        self._size = (size, size)
        self._margin = margin

    def crop(self, face: Face) -> CroppedFaceImage:
        cropped = (
            open_rgb(face.image)
            .crop(self._square_box(face.bounding_box))
            .resize(self._size, PillowImage.Resampling.BILINEAR)
        )
        return cropped_face(face, np.asarray(cropped), self._output_dir)

    def _square_box(self, box: BoundingBox) -> tuple[int, int, int, int]:
        side = max(box.width, box.height) * (1 + self._margin)
        center_x = box.x + box.width / 2
        center_y = box.y + box.height / 2
        return (
            round(center_x - side / 2),
            round(center_y - side / 2),
            round(center_x + side / 2),
            round(center_y + side / 2),
        )


class AlignedPreprocessor:
    def __init__(self, layout: FaceLayout, output_dir: Path | None = None, fallback: FacePreprocessor | None = None) -> None:
        self._layout = layout
        self._output_dir = output_dir
        self._fallback = fallback or ImagePreprocessor(layout.size, output_dir)

    def crop(self, face: Face) -> CroppedFaceImage:
        if face.landmarks is None:
            return self._fallback.crop(face)
        aligned = align_face(load_rgb(face.image), face.landmarks, self._layout)
        return cropped_face(face, aligned, self._output_dir, self._layout)


def cropped_face(
    face: Face,
    pixels: np.ndarray,
    output_dir: Path | None = None,
    layout: FaceLayout | None = None,
) -> CroppedFaceImage:
    path = None
    if output_dir is not None:
        path = output_dir / f"{re.sub(r'[^A-Za-z0-9._-]', '_', face.face_id)}.png"
        output_dir.mkdir(parents=True, exist_ok=True)
        PillowImage.fromarray(pixels).save(path)
    return CroppedFaceImage(
        face_id=face.face_id,
        image=Image(
            image_id=face.face_id,
            path=path,
            sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
            source_image_id=face.image.image_id,
            pixels=pixels,
        ),
        layout=layout,
    )
