from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class EmbedderInfo:
    name: str            # short identifier, e.g. "siglip2-so400m"
    image_kind: str      # vector_kind for image vectors, e.g. "siglip2_image"
    text_kind: str       # vector_kind for caption vectors. May differ from
                         # image space (e.g. SigLIP2 image is 1152-d but the
                         # caption channel uses MiniLM at 384-d).
    dim: int             # output dimension of image_kind (text/caption may differ)


class Embedder(Protocol):
    info: EmbedderInfo

    def load(self, device: str) -> None: ...
    def embed_image(self, images: list[Image.Image]) -> np.ndarray:
        """Image features for image_kind. Return (N, image_dim) float32 L2-normalized."""
    def embed_text(self, texts: list[str]) -> np.ndarray:
        """Text features in the IMAGE space (for cross-modal text→image retrieval).
        For SigLIP-style models this is the text encoder paired with the image
        encoder. Return (N, image_dim) float32 L2-normalized."""
    def embed_caption(self, texts: list[str]) -> np.ndarray:
        """Caption features in the CAPTION space (for text→text retrieval against
        stored caption vectors). May use a different model than embed_text — e.g.
        SigLIP2 uses a sentence-transformer here because its native text encoder
        was trained for cross-modal text→image, not text→text. Returns shape
        (N, caption_dim) float32 L2-normalized."""
