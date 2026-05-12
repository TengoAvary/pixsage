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


def test_set_and_get_camera_gps_roundtrip(tmp_path: Path):
    cat = Catalog(tmp_path / "catalog.db")
    cat.init_schema()
    img = tmp_path / "a.jpg"
    img.write_bytes(b"\x00")
    cat.upsert_photo("sha1", img, filesize=1, mtime=0.0)

    cat.set_camera_gps("sha1", latitude=-52.5, longitude=-60.4, altitude=30.5)
    got = cat.get_camera_gps("sha1")
    assert got == {"latitude": -52.5, "longitude": -60.4, "altitude": 30.5}
    cat.close()


def test_get_camera_gps_returns_none_when_absent(tmp_path: Path):
    cat = Catalog(tmp_path / "catalog.db")
    cat.init_schema()
    img = tmp_path / "a.jpg"
    img.write_bytes(b"\x00")
    cat.upsert_photo("sha1", img, filesize=1, mtime=0.0)
    assert cat.get_camera_gps("sha1") is None
    cat.close()


def test_set_camera_gps_overwrites(tmp_path: Path):
    cat = Catalog(tmp_path / "catalog.db")
    cat.init_schema()
    img = tmp_path / "a.jpg"
    img.write_bytes(b"\x00")
    cat.upsert_photo("sha1", img, filesize=1, mtime=0.0)

    cat.set_camera_gps("sha1", latitude=1.0, longitude=2.0, altitude=None)
    cat.set_camera_gps("sha1", latitude=10.0, longitude=20.0, altitude=100.0)
    got = cat.get_camera_gps("sha1")
    assert got == {"latitude": 10.0, "longitude": 20.0, "altitude": 100.0}
    cat.close()


def test_set_camera_gps_accepts_no_altitude(tmp_path: Path):
    cat = Catalog(tmp_path / "catalog.db")
    cat.init_schema()
    img = tmp_path / "a.jpg"
    img.write_bytes(b"\x00")
    cat.upsert_photo("sha1", img, filesize=1, mtime=0.0)

    cat.set_camera_gps("sha1", latitude=1.0, longitude=2.0, altitude=None)
    got = cat.get_camera_gps("sha1")
    assert got == {"latitude": 1.0, "longitude": 2.0, "altitude": None}
    cat.close()
