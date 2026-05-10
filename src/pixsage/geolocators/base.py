from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image


@dataclass(frozen=True)
class GeoPrediction:
    latitude: float
    longitude: float
    score: float


@dataclass(frozen=True)
class GeolocatorInfo:
    name: str            # short identifier, e.g. "geoclip"
    model_version: str   # e.g. "geoclip-v1"
    top_k: int           # number of predictions returned per image


class Geolocator(Protocol):
    info: GeolocatorInfo

    def load(self, device: str) -> None: ...
    def predict(self, images: list[Image.Image]) -> list[list[GeoPrediction]]:
        """One list of length info.top_k per input image, ordered by descending score."""
