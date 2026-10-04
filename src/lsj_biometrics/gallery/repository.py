from typing import Protocol

from lsj_biometrics.campaign.identity import Identity
from lsj_biometrics.campaign.template import Template
from lsj_biometrics.engine.matcher import Match

class GalleryRepository(Protocol):
    def save_identity(self, identity: Identity) -> None:
        ...

    def save_template(self, template: Template) -> None:
        ...

    def search(self, template: Template, rank: int = 5) -> list[Match]:
        ...