from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Image:
    image_id: str
    path: Path
    sha256: str | None = None