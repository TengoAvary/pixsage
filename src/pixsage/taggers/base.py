from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image


@dataclass(frozen=True)
class Tag:
    name: str
    confidence: float
    hierarchy: str | None
    source: str  # "florence2" | "ram++"


@dataclass(frozen=True)
class TagResult:
    tags: list[Tag]
    caption: str | None  # only Florence-2 produces a caption in Phase 1


class Tagger(Protocol):
    name: str
    model_version: str

    def load(self, device: str) -> None: ...
    def tag(self, image: Image.Image) -> TagResult: ...
