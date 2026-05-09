from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

from pixsage.embedders.base import Embedder, EmbedderInfo


def _seed_from(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _hashed_vector(seed_text: str, dim: int) -> np.ndarray:
    rng = np.random.default_rng(_seed_from(seed_text))
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-12
    return v


class MockEmbedder(Embedder):
    def __init__(self, dim: int = 16):
        self.info = EmbedderInfo(
            name="mock",
            image_kind="mock_image",
            text_kind="mock_text",
            dim=dim,
        )

    def load(self, device: str) -> None:
        pass

    def embed_image(self, images: list[Image.Image]) -> np.ndarray:
        # Hash the bytes of a tiny canonical thumbnail so the same image content
        # produces the same vector across runs.
        out = np.zeros((len(images), self.info.dim), dtype=np.float32)
        for i, img in enumerate(images):
            small = img.convert("RGB").resize((8, 8))
            out[i] = _hashed_vector(small.tobytes().hex(), self.info.dim)
        return out

    def embed_text(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.info.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i] = _hashed_vector(t, self.info.dim)
        return out

    def embed_caption(self, texts: list[str]) -> np.ndarray:
        # For the mock, caption space and text space share the same dim and seed
        # scheme. Real SigLIP2 uses a separate sentence-transformer here.
        out = np.zeros((len(texts), self.info.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i] = _hashed_vector("caption:" + t, self.info.dim)
        return out
