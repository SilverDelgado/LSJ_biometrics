from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from lsj_biometrics.campaign.face import CroppedFaceImage

@dataclass(frozen=True)
class Template:
    vector: tuple[float, ...] #representación para este probe
    model_name: str #modelo usado para la representación (extracción de características)
    model_version: str
    normalized: bool = False
    template_id: Optional[str] = None
    identity_id: Optional[str] = None
    face: Optional[CroppedFaceImage] = None
    created_at: Optional[datetime] = None

    @property
    def dimension(self) -> int:
        return len(self.vector)