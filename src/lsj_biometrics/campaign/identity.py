from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class Identity: #persona en el mundo real
    identity_id: str
    name: str | None = None
    created_at: datetime | None = None