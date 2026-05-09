from __future__ import annotations

from pathlib import Path

from pixsage.catalog import Catalog


def test_catalog_init_creates_schema(tmp_path: Path):
    db_path = tmp_path / "catalog.db"
    cat = Catalog(db_path)
    cat.init_schema()
    cat.close()
    assert db_path.exists()


def test_upsert_photo_inserts_row(tmp_path: Path):
    cat = Catalog(tmp_path / "c.db")
    cat.init_schema()
    cat.upsert_photo(sha256="a" * 64, path=tmp_path / "x.jpg", filesize=100, mtime=1.0)
    row = cat.get_photo("a" * 64)
    assert row is not None
    assert row["filename"] == "x.jpg"
    assert row["filesize"] == 100
    assert row["last_tagged_at"] is None
    assert row["model_versions"] is None
    cat.close()


def test_upsert_photo_updates_last_seen(tmp_path: Path):
    cat = Catalog(tmp_path / "c.db")
    cat.init_schema()
    sha = "b" * 64
    cat.upsert_photo(sha256=sha, path=tmp_path / "x.jpg", filesize=100, mtime=1.0)
    first_seen = cat.get_photo(sha)["last_seen_at"]
    cat.upsert_photo(sha256=sha, path=tmp_path / "x.jpg", filesize=100, mtime=2.0)
    second_seen = cat.get_photo(sha)["last_seen_at"]
    assert second_seen >= first_seen
    cat.close()


def test_mark_tagged_records_versions(tmp_path: Path):
    cat = Catalog(tmp_path / "c.db")
    cat.init_schema()
    sha = "c" * 64
    cat.upsert_photo(sha256=sha, path=tmp_path / "x.jpg", filesize=10, mtime=1.0)
    cat.mark_tagged(sha, model_versions={"florence2": "1.0", "ram++": "1.0"})
    row = cat.get_photo(sha)
    assert row["last_tagged_at"] is not None
    assert "florence2" in row["model_versions"]
    cat.close()


def test_needs_tagging_logic(tmp_path: Path):
    cat = Catalog(tmp_path / "c.db")
    cat.init_schema()
    sha = "d" * 64
    cat.upsert_photo(sha256=sha, path=tmp_path / "x.jpg", filesize=10, mtime=1.0)
    versions = {"florence2": "1.0", "ram++": "1.0"}
    assert cat.needs_tagging(sha, versions) is True
    cat.mark_tagged(sha, versions)
    assert cat.needs_tagging(sha, versions) is False
    assert cat.needs_tagging(sha, {"florence2": "2.0", "ram++": "1.0"}) is True
    cat.close()
