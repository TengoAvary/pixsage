from __future__ import annotations

from pathlib import Path

from pixsage.catalog import Catalog


def _seed(tmp_path: Path) -> tuple[Catalog, str]:
    cat = Catalog(tmp_path / "catalog.db")
    cat.init_schema()
    cat.upsert_photo("sha-1", tmp_path / "a.jpg", filesize=10, mtime=1.0)
    return cat, "sha-1"


def test_record_user_location(tmp_path: Path):
    cat, sha = _seed(tmp_path)
    cat.record_user_location(sha, -64.28, -56.74, "Seymour Island", "cluster:7")
    out = cat.get_user_location(sha)
    assert out is not None
    assert out["latitude"] == -64.28
    assert out["longitude"] == -56.74
    assert out["place_name"] == "Seymour Island"
    assert out["applied_via"] == "cluster:7"
    assert out["applied_at"] is not None


def test_record_user_location_overwrites(tmp_path: Path):
    cat, sha = _seed(tmp_path)
    cat.record_user_location(sha, 0.0, 0.0, "wrong", "cluster:1")
    cat.record_user_location(sha, -64.28, -56.74, "Seymour", "cluster:7")
    out = cat.get_user_location(sha)
    assert out["latitude"] == -64.28
    assert out["place_name"] == "Seymour"
    assert out["applied_via"] == "cluster:7"


def test_get_user_location_returns_none_for_unknown(tmp_path: Path):
    cat, _ = _seed(tmp_path)
    assert cat.get_user_location("missing") is None


def test_iter_user_locations(tmp_path: Path):
    cat = Catalog(tmp_path / "c.db")
    cat.init_schema()
    cat.upsert_photo("sha-a", tmp_path / "a.jpg", filesize=1, mtime=1.0)
    cat.upsert_photo("sha-b", tmp_path / "b.jpg", filesize=1, mtime=1.0)
    cat.record_user_location("sha-a", 1.0, 2.0, "A", "manual")
    cat.record_user_location("sha-b", 3.0, 4.0, None, "cluster:5")
    rows = sorted(cat.iter_user_locations(), key=lambda r: r["sha256"])
    assert [r["sha256"] for r in rows] == ["sha-a", "sha-b"]
    assert rows[1]["place_name"] is None


def test_user_location_cascade_delete(tmp_path: Path):
    cat, sha = _seed(tmp_path)
    cat.record_user_location(sha, 1.0, 2.0, "X", "manual")
    with cat._conn:  # noqa: SLF001
        cat._conn.execute("DELETE FROM photos WHERE sha256 = ?", (sha,))  # noqa: SLF001
    assert cat.get_user_location(sha) is None
