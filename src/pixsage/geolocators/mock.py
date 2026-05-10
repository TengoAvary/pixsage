from __future__ import annotations

import hashlib

from PIL import Image

from pixsage.geolocators.base import GeoPrediction, Geolocator, GeolocatorInfo


_GALLERY = [
    (51.5074, -0.1278),     # London
    (40.7128, -74.0060),    # New York
    (-33.8688, 151.2093),   # Sydney
    (35.6762, 139.6503),    # Tokyo
    (-64.0, -57.0),         # Antarctic Peninsula (rough)
]


class MockGeolocator(Geolocator):
    """Deterministic geolocator for tests. Picks an image-hash-driven ordering of
    a small fixed gallery and returns the top-K with synthetic descending scores."""

    def __init__(self, top_k: int = 3) -> None:
        self.info = GeolocatorInfo(name="mock", model_version="mock-v1", top_k=top_k)

    def load(self, device: str) -> None:
        pass

    def predict(self, images: list[Image.Image]) -> list[list[GeoPrediction]]:
        out: list[list[GeoPrediction]] = []
        for img in images:
            small = img.convert("RGB").resize((8, 8))
            digest = hashlib.sha256(small.tobytes()).digest()
            order = sorted(
                range(len(_GALLERY)),
                key=lambda i: digest[i % len(digest)],
            )
            preds: list[GeoPrediction] = []
            for rank, idx in enumerate(order[: self.info.top_k]):
                lat, lon = _GALLERY[idx]
                score = 1.0 / (rank + 1)
                preds.append(GeoPrediction(latitude=lat, longitude=lon, score=score))
            out.append(preds)
        return out
