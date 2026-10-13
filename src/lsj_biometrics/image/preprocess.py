import hashlib
from pathlib import Path

from PIL import Image as PillowImage

from lsj_biometrics.campaign.face import BoundingBox, CroppedFaceImage, Face
from lsj_biometrics.campaign.image import Image
from lsj_biometrics.image.loading import open_rgb


class ImagePreprocessor:
    def __init__(self, output_dir: Path, size: int = 112, margin: float = 0.2) -> None:
        self._output_dir = output_dir
        self._size = (size, size)
        self._margin = margin

    def crop(self, face: Face) -> CroppedFaceImage:
        cropped = (
            open_rgb(face.image.path)
            .crop(self._square_box(face.bounding_box))
            .resize(self._size, PillowImage.Resampling.BILINEAR)
        )
        path = self._output_dir / f"{face.face_id.replace(':', '_')}.png"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        cropped.save(path)
        return CroppedFaceImage(
            face_id=face.face_id,
            image=Image(
                image_id=face.face_id,
                path=path,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_image_id=face.image.image_id,
            ),
        )

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
