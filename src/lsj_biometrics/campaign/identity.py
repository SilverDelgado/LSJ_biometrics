from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class Identity:
    identity_id: str
    name: str | None = None
    created_at: datetime | None = None