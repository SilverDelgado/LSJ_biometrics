from datetime import datetime, timezone

import torch
import torch.nn.functional as F

from lsj_biometrics.campaign.face import CroppedFaceImage
from lsj_biometrics.campaign.template import Template
from lsj_biometrics.image.loading import load_rgb
from lsj_biometrics.modules.extractors.ssm.model import SSMFaceModel


class SSMFeatureExtractor:
    model_name = "ssm-face"
    model_version = "0.1.0"

    def __init__(self, model: SSMFaceModel | None = None, device: str | torch.device = "cpu") -> None:
        self._device = torch.device(device)
        self._model = (model or SSMFaceModel()).to(self._device).eval()

    @torch.no_grad()
    def extract(self, face: CroppedFaceImage) -> Template:
        size = self._model.config.image_size
        pixels = load_rgb(face.image.path, size=(size, size))
        batch = torch.tensor(pixels, dtype=torch.float32, device=self._device).permute(2, 0, 1).unsqueeze(0)
        vector = F.normalize(self._model(batch), dim=-1)[0].cpu()
        return Template(
            vector=tuple(vector.tolist()),
            model_name=self.model_name,
            model_version=self.model_version,
            normalized=True,
            created_at=datetime.now(timezone.utc),
        )
