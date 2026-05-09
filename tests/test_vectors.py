from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pixsage.vectors import VectorStore


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "vectors")


def test_append_and_load_round_trip(store: VectorStore):
    rows = [("sha1", np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            ("sha2", np.array([0.0, 1.0, 0.0], dtype=np.float32))]
    store.append("siglip2_image", rows)

    sha_array, matrix = store.load("siglip2_image")
    assert list(sha_array) == ["sha1", "sha2"]
    assert matrix.shape == (2, 3)
    assert matrix.dtype == np.float32
    np.testing.assert_array_equal(matrix[0], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(matrix[1], [0.0, 1.0, 0.0])


def test_append_replaces_existing_sha(store: VectorStore):
    store.append("siglip2_image", [("sha1", np.array([1.0, 0.0], dtype=np.float32))])
    store.append("siglip2_image", [("sha1", np.array([0.0, 1.0], dtype=np.float32))])

    sha_array, matrix = store.load("siglip2_image")
    assert list(sha_array) == ["sha1"]
    np.testing.assert_array_equal(matrix[0], [0.0, 1.0])


def test_missing_for_returns_unembedded_shas(store: VectorStore):
    store.append("siglip2_image", [
        ("sha1", np.array([1.0, 0.0], dtype=np.float32)),
        ("sha2", np.array([0.0, 1.0], dtype=np.float32)),
    ])
    missing = store.missing_for("siglip2_image", {"sha1", "sha2", "sha3", "sha4"})
    assert missing == {"sha3", "sha4"}


def test_load_empty_kind_returns_empty(store: VectorStore):
    sha_array, matrix = store.load("siglip2_image")
    assert len(sha_array) == 0
    assert matrix.shape == (0, 0)


def test_get_one_returns_vector(store: VectorStore):
    v = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    store.append("siglip2_image", [("sha1", v)])
    fetched = store.get_one("siglip2_image", "sha1")
    np.testing.assert_array_equal(fetched, v)


def test_get_one_missing_returns_none(store: VectorStore):
    assert store.get_one("siglip2_image", "missing-sha") is None


def test_created_at_recorded(store: VectorStore):
    store.append("siglip2_image", [("sha1", np.array([1.0, 0.0], dtype=np.float32))])
    ts = store.created_at("siglip2_image", "sha1")
    assert ts is not None
    # ISO 8601 ish — at least contains 'T' or '-'
    assert "T" in ts or "-" in ts


def test_append_rejects_non_float32(store: VectorStore):
    with pytest.raises(ValueError, match="must be float32"):
        store.append("siglip2_image", [("sha1", np.array([1.0, 0.0], dtype=np.float64))])


def test_append_rejects_non_1d(store: VectorStore):
    with pytest.raises(ValueError, match="must be 1-D"):
        store.append("siglip2_image", [("sha1", np.array([[1.0, 0.0]], dtype=np.float32))])
