from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class Template:
    vector: tuple[float, ...]
    model_name: str
    model_version: str
    normalized: bool = False

    @property
    def dimension(self) -> int:
        return len(self.vector)


@dataclass(frozen=True)
class GalleryTemplate:
    template_id: str
    identity_id: str
    image_id: str #could be more than one image, but for now we will assume that it is one image per template
    template: Template
    created_at: datetime | None = None