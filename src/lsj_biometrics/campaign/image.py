from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass(frozen=True)
class Image:
    image_id: str
    path: Path
    sha256: Optional[str] = None
    source_image_id: Optional[str] = None #si la imagen es recortada y viene de esta source