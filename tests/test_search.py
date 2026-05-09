from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pixsage.embedders.mock import MockEmbedder
from pixsage.search import Hit, SearchService
from pixsage.vectors import VectorStore


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "vectors")


@pytest.fixture
def embedder() -> MockEmbedder:
    return MockEmbedder(dim=8)


def _normalize(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)


def test_search_returns_top_k_by_image_when_weight_is_one(store: VectorStore, embedder: MockEmbedder):
    # Two photos: sha1 matches the query, sha2 is orthogonal.
    q_text = "leopard seal"
    q_vec = embedder.embed_text([q_text])[0]
    near = _normalize(q_vec + 0.01)                 # almost the query
    far = _normalize(np.array([1, -1, 0.5, 0, -0.3, 0.2, 1, -1], dtype=np.float32))
    store.append("mock_image", [("sha-near", near), ("sha-far", far)])

    svc = SearchService(store=store, embedder=embedder, image_kind="mock_image", text_kind="mock_text")
    svc.load()
    hits = svc.search(q_text, image_weight=1.0, top_k=5)

    assert isinstance(hits[0], Hit)
    assert hits[0].sha256 == "sha-near"
    assert hits[1].sha256 == "sha-far"


def test_search_returns_top_k_by_caption_when_weight_is_zero(store: VectorStore, embedder: MockEmbedder):
    q_text = "leopard seal"
    q_vec = embedder.embed_text([q_text])[0]
    near = _normalize(q_vec + 0.01)
    far = _normalize(np.array([1, -1, 0.5, 0, -0.3, 0.2, 1, -1], dtype=np.float32))
    store.append("mock_text", [("sha-near", near), ("sha-far", far)])

    svc = SearchService(store=store, embedder=embedder, image_kind="mock_image", text_kind="mock_text")
    svc.load()
    hits = svc.search(q_text, image_weight=0.0, top_k=5)

    assert hits[0].sha256 == "sha-near"


def test_search_handles_missing_caption_channel(store: VectorStore, embedder: MockEmbedder):
    """Photo with image vector but no caption vector should still be findable."""
    q_text = "leopard seal"
    q_vec = embedder.embed_text([q_text])[0]
    near = _normalize(q_vec + 0.01)
    store.append("mock_image", [("sha-near", near)])
    # No caption vector for sha-near.

    svc = SearchService(store=store, embedder=embedder, image_kind="mock_image", text_kind="mock_text")
    svc.load()
    hits = svc.search(q_text, image_weight=0.5, top_k=5)
    assert len(hits) == 1
    assert hits[0].sha256 == "sha-near"


def test_search_by_image_uses_pure_visual_cosine(store: VectorStore, embedder: MockEmbedder):
    a = _normalize(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    b = _normalize(np.array([0.99, 0.01, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    c = _normalize(np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    store.append("mock_image", [("a", a), ("b", b), ("c", c)])

    svc = SearchService(store=store, embedder=embedder, image_kind="mock_image", text_kind="mock_text")
    svc.load()
    hits = svc.search_by_image("a", top_k=5)

    # 'a' itself is excluded.
    shas = [h.sha256 for h in hits]
    assert shas == ["b", "c"]


def test_search_empty_store_returns_empty(store: VectorStore, embedder: MockEmbedder):
    svc = SearchService(store=store, embedder=embedder, image_kind="mock_image", text_kind="mock_text")
    svc.load()
    hits = svc.search("anything", image_weight=0.5, top_k=5)
    assert hits == []


def test_search_respects_top_k(store: VectorStore, embedder: MockEmbedder):
    q_text = "x"
    q_vec = embedder.embed_text([q_text])[0]
    rows = []
    for i in range(20):
        v = _normalize(q_vec + 0.001 * i * np.ones(8, dtype=np.float32))
        rows.append((f"sha{i}", v))
    store.append("mock_image", rows)

    svc = SearchService(store=store, embedder=embedder, image_kind="mock_image", text_kind="mock_text")
    svc.load()
    hits = svc.search(q_text, image_weight=1.0, top_k=5)
    assert len(hits) == 5
