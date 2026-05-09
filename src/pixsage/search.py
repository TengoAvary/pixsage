from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pixsage.embedders.base import Embedder
from pixsage.vectors import VectorStore


@dataclass(frozen=True)
class Hit:
    sha256: str
    score: float


class SearchService:
    """Loads vector matrices once, answers search queries via numpy.

    Combined score for a text query q at image_weight w:
        s(photo) = w * cos(q, image_vec) + (1-w) * cos(q, caption_vec)
    Photos missing a channel score that channel as 0.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        image_kind: str,
        text_kind: str,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.image_kind = image_kind
        self.text_kind = text_kind

        self._img_shas: np.ndarray = np.array([], dtype=object)
        self._img_matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._txt_shas: np.ndarray = np.array([], dtype=object)
        self._txt_matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._idx_img: dict[str, int] = {}
        self._idx_txt: dict[str, int] = {}

    def load(self) -> None:
        self._img_shas, self._img_matrix = self.store.load(self.image_kind)
        self._txt_shas, self._txt_matrix = self.store.load(self.text_kind)
        self._idx_img = {s: i for i, s in enumerate(self._img_shas.tolist())}
        self._idx_txt = {s: i for i, s in enumerate(self._txt_shas.tolist())}

    def search(self, query: str, image_weight: float, top_k: int) -> list[Hit]:
        if self._img_matrix.size == 0 and self._txt_matrix.size == 0:
            return []

        # Two encoders, two query vectors. embed_text is the cross-modal text
        # encoder paired with the image encoder (e.g. SigLIP2 text → image
        # space). embed_caption is the document encoder for caption text→text
        # retrieval (e.g. MiniLM). They live in different vector spaces and
        # different dimensions; each scores against its own matrix.
        visual_q = self.embedder.embed_text([query])[0]
        caption_q = self.embedder.embed_caption([query])[0]

        img_scores = (
            self._img_matrix @ visual_q if self._img_matrix.size else np.zeros(0, dtype=np.float32)
        )
        txt_scores = (
            self._txt_matrix @ caption_q if self._txt_matrix.size else np.zeros(0, dtype=np.float32)
        )

        all_shas = set(self._idx_img.keys()) | set(self._idx_txt.keys())
        hits: list[Hit] = []
        for sha in all_shas:
            i_img = self._idx_img.get(sha)
            i_txt = self._idx_txt.get(sha)
            si = float(img_scores[i_img]) if i_img is not None else 0.0
            st = float(txt_scores[i_txt]) if i_txt is not None else 0.0
            score = image_weight * si + (1.0 - image_weight) * st
            hits.append(Hit(sha256=sha, score=score))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def search_by_image(self, sha256: str, top_k: int) -> list[Hit]:
        if self._img_matrix.size == 0:
            return []
        idx = self._idx_img.get(sha256)
        if idx is None:
            return []
        q = self._img_matrix[idx]
        scores = self._img_matrix @ q
        ranked = np.argsort(-scores)
        hits: list[Hit] = []
        for j in ranked:
            sha = self._img_shas[j]
            if sha == sha256:
                continue
            hits.append(Hit(sha256=str(sha), score=float(scores[j])))
            if len(hits) >= top_k:
                break
        return hits
