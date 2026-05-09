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
