from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class EmbedderInfo:
    name: str            # short identifier, e.g. "siglip2-so400m"
    image_kind: str      # vector_kind for image vectors, e.g. "siglip2_image"
    text_kind: str       # vector_kind for caption/text vectors, e.g. "siglip2_caption"
    dim: int             # output dimension (image and text share dim for SigLIP-style)


class Embedder(Protocol):
    info: EmbedderInfo

    def load(self, device: str) -> None: ...
    def embed_image(self, images: list[Image.Image]) -> np.ndarray:
        """Return (N, dim) float32 L2-normalized."""
    def embed_text(self, texts: list[str]) -> np.ndarray:
        """Return (N, dim) float32 L2-normalized."""
