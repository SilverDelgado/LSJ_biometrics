from dataclasses import dataclass
from typing import Protocol

from lsj_biometrics.campaign.template import Template

@dataclass(frozen=True)
class Match:
    identity_id: str
    template_id: str
    score: float

class Matcher(Protocol):
    def compare(self, first: Template, second: Template) -> float:
        ...