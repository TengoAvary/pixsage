from __future__ import annotations

from pathlib import Path

from pixsage.catalog import Catalog


def test_init_schema_adds_exif_gps_columns(tmp_path: Path):
    cat = Catalog(tmp_path / "catalog.db")
    cat.init_schema()
    cur = cat._conn.execute("PRAGMA table_info(photos)")
    columns = {row["name"] for row in cur.fetchall()}
    assert "exif_latitude" in columns
    assert "exif_longitude" in columns
    assert "exif_altitude" in columns
    cat.close()


def test_init_schema_is_idempotent_for_exif_columns(tmp_path: Path):
    db = tmp_path / "catalog.db"
    Catalog(db).init_schema()
    # Second init must not raise "duplicate column" error.
    Catalog(db).init_schema()
