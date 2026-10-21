import hashlib
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from PIL import Image as PillowImage

from lsj_biometrics.campaign.face import CroppedFaceImage
from lsj_biometrics.image.loading import load_rgb


class FileCropRepository:
    def __init__(self, root: Path) -> None:
        self._root = root

    def save(self, face: CroppedFaceImage) -> CroppedFaceImage:
        pixels = load_rgb(face.image)
        sha256 = hashlib.sha256(pixels.tobytes()).hexdigest()
        path = self._root / sha256[:2] / f"{sha256}.png"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(f"{sha256}.{uuid4().hex}.partial")
            PillowImage.fromarray(pixels).save(partial, format="PNG")
            partial.replace(path)
        return replace(face, image=replace(face.image, path=path, sha256=sha256, pixels=pixels))
