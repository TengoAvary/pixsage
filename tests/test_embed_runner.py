from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixsage.catalog import Catalog
from pixsage.embed_runner import EmbedRunner
from pixsage.embedders.mock import MockEmbedder
from pixsage.vectors import VectorStore


@pytest.fixture
def catalog(tmp_path: Path) -> Catalog:
    cat = Catalog(tmp_path / "catalog.db")
    cat.init_schema()
    return cat


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "vectors")


def _seed_photo(catalog: Catalog, sha: str, img_path: Path, caption: str | None = None) -> None:
    Image.new("RGB", (32, 32), color="red").save(img_path)
    catalog.upsert_photo(sha, img_path, filesize=img_path.stat().st_size, mtime=img_path.stat().st_mtime)
    if caption is not None:
        catalog.record_caption(sha, caption)


def test_runner_embeds_image_and_caption(catalog: Catalog, store: VectorStore, tmp_path: Path):
    _seed_photo(catalog, "sha1", tmp_path / "a.jpg", caption="a leopard seal")

    runner = EmbedRunner(catalog=catalog, vectors=store, embedder=MockEmbedder(dim=8))
    runner.run()

    sha_img, mat_img = store.load("mock_image")
    sha_txt, mat_txt = store.load("mock_text")
    assert list(sha_img) == ["sha1"]
    assert list(sha_txt) == ["sha1"]
    assert mat_img.shape == (1, 8)
    assert mat_txt.shape == (1, 8)


def test_runner_skips_caption_when_absent(catalog: Catalog, store: VectorStore, tmp_path: Path):
    _seed_photo(catalog, "sha1", tmp_path / "a.jpg", caption=None)

    runner = EmbedRunner(catalog=catalog, vectors=store, embedder=MockEmbedder(dim=8))
    runner.run()

    sha_img, _ = store.load("mock_image")
    sha_txt, _ = store.load("mock_text")
    assert list(sha_img) == ["sha1"]
    assert list(sha_txt) == []


def test_runner_skips_already_embedded(catalog: Catalog, store: VectorStore, tmp_path: Path):
    _seed_photo(catalog, "sha1", tmp_path / "a.jpg", caption="x")
    embedder = MockEmbedder(dim=8)

    EmbedRunner(catalog=catalog, vectors=store, embedder=embedder).run()
    # Track call count via wrapping.
    calls = {"image": 0, "text": 0}
    real_image, real_text = embedder.embed_image, embedder.embed_text
    embedder.embed_image = lambda imgs: (calls.__setitem__("image", calls["image"] + 1), real_image(imgs))[1]
    embedder.embed_text = lambda txts: (calls.__setitem__("text", calls["text"] + 1), real_text(txts))[1]

    EmbedRunner(catalog=catalog, vectors=store, embedder=embedder).run()
    assert calls == {"image": 0, "text": 0}


def test_runner_reembeds_on_caption_staleness(catalog: Catalog, store: VectorStore, tmp_path: Path):
    _seed_photo(catalog, "sha1", tmp_path / "a.jpg", caption="first")
    EmbedRunner(catalog=catalog, vectors=store, embedder=MockEmbedder(dim=8)).run()
    first_vec = store.get_one("mock_text", "sha1")

    time.sleep(0.05)
    catalog.record_caption("sha1", "completely different caption")
    EmbedRunner(catalog=catalog, vectors=store, embedder=MockEmbedder(dim=8)).run()
    second_vec = store.get_one("mock_text", "sha1")

    assert not np.array_equal(first_vec, second_vec)


def test_runner_force_reembeds_everything(catalog: Catalog, store: VectorStore, tmp_path: Path):
    _seed_photo(catalog, "sha1", tmp_path / "a.jpg", caption="first")
    embedder = MockEmbedder(dim=8)
    EmbedRunner(catalog=catalog, vectors=store, embedder=embedder).run()

    calls = {"image": 0}
    real_image = embedder.embed_image
    embedder.embed_image = lambda imgs: (calls.__setitem__("image", calls["image"] + 1), real_image(imgs))[1]

    EmbedRunner(catalog=catalog, vectors=store, embedder=embedder, force=True).run()
    assert calls["image"] >= 1


def test_runner_marks_decode_errors(catalog: Catalog, store: VectorStore, tmp_path: Path):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not an image")
    catalog.upsert_photo("sha-bad", bad, filesize=bad.stat().st_size, mtime=bad.stat().st_mtime)

    runner = EmbedRunner(catalog=catalog, vectors=store, embedder=MockEmbedder(dim=8))
    runner.run()

    row = catalog.get_photo("sha-bad")
    assert row["error_reason"] is not None
