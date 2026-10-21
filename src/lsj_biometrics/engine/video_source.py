from typing import Protocol

from lsj_biometrics.campaign.frame import Frame

class StreamEnded(Exception):
    pass

class VideoSource(Protocol):
    def poll(self) -> Frame | None:
        ...

    def close(self) -> None:
        ...