from __future__ import annotations

from pathlib import Path

import pytest

from pixsage.catalog import Catalog


@pytest.fixture
def catalog(tmp_path: Path) -> Catalog:
    cat = Catalog(tmp_path / "catalog.db")
    cat.init_schema()
    return cat


def test_caption_columns_exist(catalog: Catalog):
    cur = catalog._conn.execute("PRAGMA table_info(photos)")
    cols = {row["name"] for row in cur.fetchall()}
    assert "caption" in cols
    assert "caption_updated_at" in cols


def test_record_caption_sets_text_and_timestamp(catalog: Catalog, tmp_path: Path):
    catalog.upsert_photo("sha1", tmp_path / "a.jpg", filesize=10, mtime=1.0)
    catalog.record_caption("sha1", "a leopard seal on ice")

    row = catalog.get_photo("sha1")
    assert row["caption"] == "a leopard seal on ice"
    assert row["caption_updated_at"] is not None  # ISO timestamp


def test_record_caption_updates_timestamp_on_change(catalog: Catalog, tmp_path: Path):
    catalog.upsert_photo("sha1", tmp_path / "a.jpg", filesize=10, mtime=1.0)
    catalog.record_caption("sha1", "first")
    first_ts = catalog.get_photo("sha1")["caption_updated_at"]

    catalog.record_caption("sha1", "second")
    second_ts = catalog.get_photo("sha1")["caption_updated_at"]
    assert second_ts > first_ts


def test_record_caption_none_clears_caption(catalog: Catalog, tmp_path: Path):
    catalog.upsert_photo("sha1", tmp_path / "a.jpg", filesize=10, mtime=1.0)
    catalog.record_caption("sha1", "text")
    catalog.record_caption("sha1", None)
    row = catalog.get_photo("sha1")
    assert row["caption"] is None


def test_record_caption_nonexistent_sha_is_silent(catalog: Catalog):
    # Calling record_caption on a sha that isn't in photos should silently
    # update zero rows (UPDATE semantics), not raise.
    catalog.record_caption("nonexistent-sha", "text")
    assert catalog.get_photo("nonexistent-sha") is None


def test_iter_photos_for_embedding_returns_all(catalog: Catalog, tmp_path: Path):
    catalog.upsert_photo("sha1", tmp_path / "a.jpg", filesize=10, mtime=1.0)
    catalog.upsert_photo("sha2", tmp_path / "b.jpg", filesize=20, mtime=2.0)
    catalog.record_caption("sha1", "caption a")
    # sha2 has no caption

    rows = list(catalog.iter_photos_for_embedding())
    assert len(rows) == 2
    by_sha = {r["sha256"]: r for r in rows}
    assert by_sha["sha1"]["caption"] == "caption a"
    assert by_sha["sha2"]["caption"] is None
    assert by_sha["sha1"]["current_path"] == str(tmp_path / "a.jpg")


def test_iter_photos_for_embedding_skips_errored(catalog: Catalog, tmp_path: Path):
    catalog.upsert_photo("sha1", tmp_path / "a.jpg", filesize=10, mtime=1.0)
    catalog.upsert_photo("sha2", tmp_path / "b.jpg", filesize=20, mtime=2.0)
    catalog.mark_error("sha2", "decode failed")

    rows = list(catalog.iter_photos_for_embedding())
    assert {r["sha256"] for r in rows} == {"sha1"}


def test_clear_error_resets_error_reason(catalog: Catalog, tmp_path: Path):
    catalog.upsert_photo("sha1", tmp_path / "a.jpg", filesize=10, mtime=1.0)
    catalog.mark_error("sha1", "boom")
    assert catalog.get_photo("sha1")["error_reason"] == "boom"
    catalog.clear_error("sha1")
    assert catalog.get_photo("sha1")["error_reason"] is None
