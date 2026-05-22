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


def test_extend_writes_partfile_without_rewriting(store: VectorStore):
    """Each extend() is its own part-file; earlier files are never touched."""
    store.extend("siglip2_image", [("sha1", np.array([1.0, 0.0], dtype=np.float32))])
    parts_after_first = sorted((store.root / "siglip2_image").glob("*.parquet"))
    first = parts_after_first[0]
    first_mtime = first.stat().st_mtime_ns

    store.extend("siglip2_image", [("sha2", np.array([0.0, 1.0], dtype=np.float32))])
    parts_after_second = sorted((store.root / "siglip2_image").glob("*.parquet"))

    assert len(parts_after_first) == 1
    assert len(parts_after_second) == 2
    # The first part-file was not rewritten.
    assert first.stat().st_mtime_ns == first_mtime
    # Legacy single file is never created by extend.
    assert not (store.root / "siglip2_image.parquet").exists()

    sha_array, matrix = store.load("siglip2_image")
    assert sorted(sha_array) == ["sha1", "sha2"]
    assert matrix.shape == (2, 2)


def test_extend_merges_with_legacy_file_last_write_wins(store: VectorStore):
    """A legacy single file plus newer part-files merge; part-files win."""
    store.append("siglip2_image", [("sha1", np.array([1.0, 1.0], dtype=np.float32))])
    assert (store.root / "siglip2_image.parquet").exists()  # legacy

    store.extend("siglip2_image", [("sha1", np.array([9.0, 9.0], dtype=np.float32)),
                                   ("sha2", np.array([2.0, 2.0], dtype=np.float32))])

    np.testing.assert_array_equal(store.get_one("siglip2_image", "sha1"), [9.0, 9.0])
    np.testing.assert_array_equal(store.get_one("siglip2_image", "sha2"), [2.0, 2.0])


def test_load_preserves_first_seen_order_with_last_write_value(store: VectorStore):
    """load() returns shas in first-appearance order but the last-written
    vector value — matching the merge semantics of the original dict path."""
    store.append("siglip2_image", [
        ("sha1", np.array([1.0, 1.0], dtype=np.float32)),
        ("sha2", np.array([2.0, 2.0], dtype=np.float32)),
    ])  # legacy file
    store.extend("siglip2_image", [
        ("sha1", np.array([9.0, 9.0], dtype=np.float32)),  # overrides sha1 value
        ("sha3", np.array([3.0, 3.0], dtype=np.float32)),  # new
    ])

    shas, matrix = store.load("siglip2_image")
    # sha1 keeps its first-seen position; sha3 appended after sha2.
    assert list(shas) == ["sha1", "sha2", "sha3"]
    np.testing.assert_array_equal(matrix[0], [9.0, 9.0])  # last write wins
    np.testing.assert_array_equal(matrix[1], [2.0, 2.0])
    np.testing.assert_array_equal(matrix[2], [3.0, 3.0])


def test_load_wide_vectors_roundtrip(store: VectorStore):
    """Multi-row, wide (D>1) matrix loads with correct shape and values —
    guards the flat-buffer reshape in the zero-copy load path."""
    rng = np.random.default_rng(0)
    rows = [(f"sha{i}", rng.standard_normal(1152).astype(np.float32)) for i in range(50)]
    store.extend("siglip2_image", rows)

    shas, matrix = store.load("siglip2_image")
    assert matrix.shape == (50, 1152)
    assert matrix.dtype == np.float32
    by_sha = {s: matrix[i] for i, s in enumerate(shas)}
    for sha, vec in rows:
        np.testing.assert_array_equal(by_sha[sha], vec)


def test_extend_write_cost_is_constant_not_quadratic(store: VectorStore):
    """Writing N batches must not rewrite prior data — total bytes written
    grows linearly with rows, not with rows^2. Proxy: every part-file holds
    only its own batch's rows, independent of how many came before."""
    for i in range(20):
        store.extend("siglip2_image", [(f"sha{i}", np.array([float(i), 0.0], dtype=np.float32))])

    import pyarrow.parquet as pq
    parts = sorted((store.root / "siglip2_image").glob("*.parquet"))
    assert len(parts) == 20
    # Each part-file is exactly one row — no accumulation/rewrite.
    for p in parts:
        assert pq.read_table(p).num_rows == 1

    shas, matrix = store.load("siglip2_image")
    assert len(shas) == 20
