from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from pixsage.embedders.base import Embedder, EmbedderInfo


class SigLIP2Embedder(Embedder):
    """Wraps google/siglip2-so400m-patch14-384 for image+visual-query encoding,
    plus sentence-transformers/all-MiniLM-L6-v2 for caption text→text retrieval.

    Loads both lazily in `load()`. SigLIP2 is fp16 on CUDA, fp32 otherwise. The
    MiniLM caption encoder is small enough (~80MB) to always run in fp32.

    Why two encoders: SigLIP2's text encoder is trained to align text with
    images (cross-modal), so it works well for `embed_text` (encoding a query
    that we want to score against image vectors). It does NOT work well for
    text-to-text retrieval — diagnostic on the demo corpus showed inter-caption
    cosines clustering at 0.45 and the literal-camera photos ranking last for
    a "camera" query. MiniLM is a sentence-transformer trained for semantic
    text similarity; it discriminates captions cleanly.
    """

    MODEL_ID = "google/siglip2-so400m-patch14-384"
    CAPTION_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    CAPTION_DIM = 384

    def __init__(self) -> None:
        self.info = EmbedderInfo(
            name="siglip2-so400m-patch14-384",
            image_kind="siglip2_image",
            text_kind="minilm_caption",
            dim=1152,  # image_kind dim; caption_kind is CAPTION_DIM (384)
        )
        self._model: Any | None = None
        self._processor: Any | None = None
        self._caption_model: Any | None = None
        self._device: str = "cpu"
        self._dtype: Any = None

    def load(self, device: str) -> None:
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModel, AutoProcessor

        self._device = device
        self._dtype = torch.float16 if device == "cuda" else torch.float32
        # SigLIP2 ships a fast tokenizer (tokenizer.json) and no sentencepiece
        # vocab. Force use_fast=True so AutoProcessor doesn't fall back to the
        # legacy SiglipTokenizer (which would crash looking for a vocab_file).
        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID, use_fast=True)
        model = AutoModel.from_pretrained(self.MODEL_ID, torch_dtype=self._dtype)
        model.to(device).eval()
        self._model = model
        # Verify dim matches what we declared.
        actual_dim = int(model.config.text_config.hidden_size)
        if actual_dim != self.info.dim:
            self.info = EmbedderInfo(
                name=self.info.name,
                image_kind=self.info.image_kind,
                text_kind=self.info.text_kind,
                dim=actual_dim,
            )
        # Caption text encoder. Use the same device; MiniLM is tiny.
        self._caption_model = SentenceTransformer(self.CAPTION_MODEL_ID, device=device)

    def embed_image(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        assert self._model is not None and self._processor is not None
        inputs = self._processor(images=images, return_tensors="pt").to(self._device)
        # SigLIP2 processor returns float32 pixel_values; cast to model dtype.
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._dtype)
        with torch.inference_mode():
            features = self._model.get_image_features(**inputs)
        features = torch.nn.functional.normalize(features, dim=-1)
        return features.float().cpu().numpy()

    def embed_text(self, texts: list[str]) -> np.ndarray:
        import torch

        assert self._model is not None and self._processor is not None
        # SigLIP2 text encoder has max_position_embeddings=64. Truncate short
        # queries fit within this naturally; longer phrases get clipped.
        inputs = self._processor(
            text=texts,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(self._device)
        with torch.inference_mode():
            features = self._model.get_text_features(**inputs)
        features = torch.nn.functional.normalize(features, dim=-1)
        return features.float().cpu().numpy()

    def embed_caption(self, texts: list[str]) -> np.ndarray:
        assert self._caption_model is not None
        vecs = self._caption_model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)
