from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient


def _seed_root(tmp_path: Path) -> Path:
    """Build a photo_root with two photos, mock embeddings already populated."""
    from pixsage.catalog import Catalog
    from pixsage.embedders.mock import MockEmbedder
    from pixsage.vectors import VectorStore

    root = tmp_path / "photos"
    root.mkdir()
    photoindex = root / ".photoindex"
    photoindex.mkdir()

    a, b = root / "a.jpg", root / "b.jpg"
    Image.new("RGB", (64, 64), color="red").save(a)
    Image.new("RGB", (64, 64), color="blue").save(b)

    cat = Catalog(photoindex / "catalog.db")
    cat.init_schema()
    cat.upsert_photo("sha-a", a, filesize=a.stat().st_size, mtime=a.stat().st_mtime)
    cat.upsert_photo("sha-b", b, filesize=b.stat().st_size, mtime=b.stat().st_mtime)
    cat.record_caption("sha-a", "a red square")
    cat.record_caption("sha-b", "a blue square")

    embedder = MockEmbedder()  # must match build_app's default dim (16)
    vec_a_img = embedder.embed_image([Image.open(a)])[0]
    vec_b_img = embedder.embed_image([Image.open(b)])[0]
    vec_a_txt = embedder.embed_text(["a red square"])[0]
    vec_b_txt = embedder.embed_text(["a blue square"])[0]

    store = VectorStore(photoindex / "vectors")
    store.append("mock_image", [("sha-a", vec_a_img), ("sha-b", vec_b_img)])
    store.append("mock_text", [("sha-a", vec_a_txt), ("sha-b", vec_b_txt)])

    cat.close()
    return root


def test_search_returns_results_html(tmp_path: Path):
    from pixsage.web.app import build_app

    root = _seed_root(tmp_path)
    app = build_app(photo_root=root, embedder_name="mock")
    with TestClient(app) as client:
        r = client.post("/search", data={"q": "a red square", "image_weight": "0.0"})
        assert r.status_code == 200
        assert "sha-a" in r.text or "sha-b" in r.text
        assert "<article" in r.text or 'class="card"' in r.text


def test_search_empty_query_returns_empty_results(tmp_path: Path):
    from pixsage.web.app import build_app

    root = _seed_root(tmp_path)
    app = build_app(photo_root=root, embedder_name="mock")
    with TestClient(app) as client:
        r = client.post("/search", data={"q": "", "image_weight": "0.5"})
        assert r.status_code == 200
        # Empty results section, no cards
        assert "card" not in r.text.lower() or 'class="card"' not in r.text


def test_thumb_route_returns_jpeg(tmp_path: Path):
    from pixsage.web.app import build_app

    root = _seed_root(tmp_path)
    app = build_app(photo_root=root, embedder_name="mock")
    with TestClient(app) as client:
        r = client.get("/thumb/sha-a?size=small")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"
        assert len(r.content) > 0


def test_thumb_route_404_for_missing_sha(tmp_path: Path):
    from pixsage.web.app import build_app

    root = _seed_root(tmp_path)
    app = build_app(photo_root=root, embedder_name="mock")
    with TestClient(app) as client:
        r = client.get("/thumb/nonexistent-sha?size=small")
        assert r.status_code == 404
