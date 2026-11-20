from dataclasses import dataclass


@dataclass(frozen=True)
class Match:
    identity_id: str
    template_id: str
    score: float
