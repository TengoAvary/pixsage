from __future__ import annotations

from pathlib import Path

from pixsage.catalog import Catalog
from pixsage.geolocators.base import GeoPrediction


def _make_catalog(tmp_path: Path) -> Catalog:
    cat = Catalog(tmp_path / "catalog.db")
    cat.init_schema()
    return cat


def _seed_photo(cat: Catalog, sha: str, tmp_path: Path) -> None:
    p = tmp_path / f"{sha}.jpg"
    p.write_bytes(b"fake")
    cat.upsert_photo(sha, p, filesize=4, mtime=p.stat().st_mtime)


def test_record_and_get_geo_predictions(tmp_path: Path):
    cat = _make_catalog(tmp_path)
    _seed_photo(cat, "sha1", tmp_path)
    preds = [
        GeoPrediction(latitude=51.5, longitude=-0.1, score=0.7),
        GeoPrediction(latitude=40.7, longitude=-74.0, score=0.2),
    ]
    cat.record_geo_predictions("sha1", "geoclip", preds)

    out = cat.get_geo_predictions("sha1", "geoclip")
    assert len(out) == 2
    assert out[0].latitude == 51.5
    assert out[0].score == 0.7
    assert out[1].latitude == 40.7


def test_record_replaces_prior_predictions_for_same_model(tmp_path: Path):
    cat = _make_catalog(tmp_path)
    _seed_photo(cat, "sha1", tmp_path)
    cat.record_geo_predictions(
        "sha1", "geoclip", [GeoPrediction(latitude=1.0, longitude=2.0, score=0.5)]
    )
    cat.record_geo_predictions(
        "sha1",
        "geoclip",
        [
            GeoPrediction(latitude=3.0, longitude=4.0, score=0.8),
            GeoPrediction(latitude=5.0, longitude=6.0, score=0.1),
        ],
    )
    out = cat.get_geo_predictions("sha1", "geoclip")
    assert len(out) == 2
    assert out[0].latitude == 3.0


def test_predictions_isolated_per_model(tmp_path: Path):
    cat = _make_catalog(tmp_path)
    _seed_photo(cat, "sha1", tmp_path)
    cat.record_geo_predictions(
        "sha1", "geoclip", [GeoPrediction(latitude=1.0, longitude=2.0, score=0.5)]
    )
    cat.record_geo_predictions(
        "sha1", "other", [GeoPrediction(latitude=3.0, longitude=4.0, score=0.7)]
    )
    assert cat.get_geo_predictions("sha1", "geoclip")[0].latitude == 1.0
    assert cat.get_geo_predictions("sha1", "other")[0].latitude == 3.0


def test_get_returns_empty_when_no_predictions(tmp_path: Path):
    cat = _make_catalog(tmp_path)
    _seed_photo(cat, "sha1", tmp_path)
    assert cat.get_geo_predictions("sha1", "geoclip") == []


def test_iter_photos_for_geolocation_excludes_errored_by_default(tmp_path: Path):
    cat = _make_catalog(tmp_path)
    _seed_photo(cat, "sha1", tmp_path)
    _seed_photo(cat, "sha2", tmp_path)
    cat.mark_error("sha1", "boom")

    rows_default = list(cat.iter_photos_for_geolocation())
    rows_with_err = list(cat.iter_photos_for_geolocation(include_errored=True))
    assert {r["sha256"] for r in rows_default} == {"sha2"}
    assert {r["sha256"] for r in rows_with_err} == {"sha1", "sha2"}


def test_predictions_cascade_delete(tmp_path: Path):
    """Deleting a photo row should cascade-delete its geo_predictions."""
    cat = _make_catalog(tmp_path)
    _seed_photo(cat, "sha1", tmp_path)
    cat.record_geo_predictions(
        "sha1", "geoclip", [GeoPrediction(latitude=1.0, longitude=2.0, score=0.5)]
    )
    with cat._conn:  # noqa: SLF001
        cat._conn.execute("DELETE FROM photos WHERE sha256 = ?", ("sha1",))  # noqa: SLF001
    assert cat.get_geo_predictions("sha1", "geoclip") == []
