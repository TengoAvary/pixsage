from __future__ import annotations

from pathlib import Path

from pixsage.catalog import Catalog
from pixsage.taggers.base import Tag


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


def test_record_tags_inserts_rows(tmp_path: Path):
    cat = Catalog(tmp_path / "c.db")
    cat.init_schema()
    sha = "e" * 64
    cat.upsert_photo(sha256=sha, path=tmp_path / "x.jpg", filesize=1, mtime=1.0)
    tags = [
        Tag("penguin", 1.0, "Wildlife|Bird|Penguin", "florence2"),
        Tag("bird", 0.9, None, "ram++"),
    ]
    cat.record_tags(sha, tags)
    stored = cat.get_tags(sha)
    assert {t.name for t in stored} == {"penguin", "bird"}
    cat.close()


def test_record_tags_idempotent(tmp_path: Path):
    cat = Catalog(tmp_path / "c.db")
    cat.init_schema()
    sha = "f" * 64
    cat.upsert_photo(sha256=sha, path=tmp_path / "x.jpg", filesize=1, mtime=1.0)
    tags = [Tag("penguin", 1.0, None, "florence2")]
    cat.record_tags(sha, tags)
    cat.record_tags(sha, tags)
    assert len(cat.get_tags(sha)) == 1
    cat.close()


def test_user_rejected_flagging(tmp_path: Path):
    cat = Catalog(tmp_path / "c.db")
    cat.init_schema()
    sha = "0" * 64
    cat.upsert_photo(sha256=sha, path=tmp_path / "x.jpg", filesize=1, mtime=1.0)
    cat.record_tags(sha, [
        Tag("penguin", 1.0, None, "florence2"),
        Tag("ice", 0.8, None, "florence2"),
    ])
    # Pretend the user removed "ice" from XMP. We pass the surviving set:
    cat.flag_user_rejections(sha, surviving_xmp_tags={"penguin"})
    rejected = cat.get_user_rejected(sha)
    assert rejected == {("ice", "florence2")}
    not_rejected = {t.name for t in cat.get_tags(sha) if not cat.is_user_rejected(sha, t.name, t.source)}
    assert not_rejected == {"penguin"}
    cat.close()


def test_user_rejected_persists_across_record(tmp_path: Path):
    cat = Catalog(tmp_path / "c.db")
    cat.init_schema()
    sha = "1" * 64
    cat.upsert_photo(sha256=sha, path=tmp_path / "x.jpg", filesize=1, mtime=1.0)
    cat.record_tags(sha, [Tag("ice", 1.0, None, "florence2")])
    cat.flag_user_rejections(sha, surviving_xmp_tags=set())
    # Re-record: should NOT clear the rejection flag.
    cat.record_tags(sha, [Tag("ice", 1.0, None, "florence2")])
    assert cat.is_user_rejected(sha, "ice", "florence2") is True
    cat.close()


def test_runs_table_records_run(tmp_path: Path):
    cat = Catalog(tmp_path / "c.db")
    cat.init_schema()
    run_id = cat.start_run(config_hash="abc", model_versions={"florence2": "1.0"})
    assert isinstance(run_id, int)
    cat.finish_run(run_id, processed=5, skipped=2, errored=0)
    runs = cat.list_runs()
    assert len(runs) == 1
    assert runs[0]["photos_processed"] == 5
    assert runs[0]["photos_errored"] == 0
    cat.close()
