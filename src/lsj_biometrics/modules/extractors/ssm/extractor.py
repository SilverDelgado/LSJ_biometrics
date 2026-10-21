from datetime import datetime, timezone
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from lsj_biometrics.campaign.face import CroppedFaceImage, FaceLayout, Landmarks
from lsj_biometrics.campaign.template import Template
from lsj_biometrics.image.loading import load_rgb
from lsj_biometrics.modules.extractors.ssm.model import SSMFaceModel


class SSMFeatureExtractor:
    def __init__(self, model: SSMFaceModel, device: str | torch.device = "cpu", batch_size: int = 32) -> None:
        self._device = torch.device(device)
        self._model = model.to(self._device).eval()
        self._batch_size = batch_size
        config = model.config
        self.model_name = config.name
        self.model_version = config.version
        self.layout = FaceLayout(size=config.image_size, landmarks=Landmarks(*config.reference_landmarks))

    def extract(self, face: CroppedFaceImage) -> Template:
        return self.extract_batch([face])[0]

    def extract_batch(self, faces: Sequence[CroppedFaceImage]) -> list[Template]:
        return [
            template
            for start in range(0, len(faces), self._batch_size)
            for template in self._extract_chunk(faces[start : start + self._batch_size])
        ]

    @torch.inference_mode()
    def _extract_chunk(self, faces: Sequence[CroppedFaceImage]) -> list[Template]:
        pixels = np.stack([self._pixels(face) for face in faces])
        batch = torch.from_numpy(pixels).to(self._device).permute(0, 3, 1, 2).float()
        vectors = F.normalize(self._model(batch), dim=-1).cpu()
        created_at = datetime.now(timezone.utc)
        return [
            Template(
                vector=tuple(vector.tolist()),
                model_name=self.model_name,
                model_version=self.model_version,
                normalized=True,
                created_at=created_at,
            )
            for vector in vectors
        ]

    def _pixels(self, face: CroppedFaceImage) -> np.ndarray:
        if face.layout is not None and face.layout != self.layout:
            raise ValueError(
                f"face {face.face_id} was aligned for a different layout than {self.model_name}:{self.model_version}"
            )
        return load_rgb(face.image, size=(self.layout.size, self.layout.size))
