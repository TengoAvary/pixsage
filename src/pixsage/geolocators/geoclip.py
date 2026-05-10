from __future__ import annotations

from typing import Any

from PIL import Image

from pixsage.geolocators.base import GeoPrediction, Geolocator, GeolocatorInfo


class GeoCLIPGeolocator(Geolocator):
    """Wraps github.com/VicenteVivan/geo-clip.

    The published `predict()` only accepts a file path (it calls Image.open
    internally). We bypass that by handing a pre-loaded PIL.Image directly to
    `image_encoder.preprocess_image`, which lets pixsage's load_image() handle
    raw decode + EXIF orientation upstream and avoids a temp-file write per
    photo.
    """

    def __init__(self, top_k: int = 5) -> None:
        self.info = GeolocatorInfo(
            name="geoclip",
            model_version="geoclip-v1",
            top_k=top_k,
        )
        self._model: Any | None = None
        self._device: str = "cpu"
        self._gps_gallery: Any | None = None

    def load(self, device: str) -> None:
        from geoclip import GeoCLIP
        self._device = device
        model = GeoCLIP()
        model.to(device)
        model.eval()
        self._model = model
        self._gps_gallery = model.gps_gallery.to(device)

    def predict(self, images: list[Image.Image]) -> list[list[GeoPrediction]]:
        import torch

        assert self._model is not None and self._gps_gallery is not None
        out: list[list[GeoPrediction]] = []
        for img in images:
            tensor = self._model.image_encoder.preprocess_image(img.convert("RGB"))
            tensor = tensor.to(self._device)
            with torch.inference_mode():
                logits = self._model.forward(tensor, self._gps_gallery)
                probs = logits.softmax(dim=-1)[0].cpu()
            top = torch.topk(probs, self.info.top_k)
            preds: list[GeoPrediction] = []
            for score, idx in zip(top.values.tolist(), top.indices.tolist()):
                lat, lon = self._gps_gallery[idx].cpu().tolist()
                preds.append(GeoPrediction(
                    latitude=float(lat),
                    longitude=float(lon),
                    score=float(score),
                ))
            out.append(preds)
        return out
