from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from pixsage.catalog import Catalog
from pixsage.geo_runner import GeoRunner
from pixsage.geolocators.mock import MockGeolocator


@pytest.fixture
def catalog(tmp_path: Path) -> Catalog:
    cat = Catalog(tmp_path / "catalog.db")
    cat.init_schema()
    return cat


def _seed_photo(cat: Catalog, sha: str, tmp_path: Path, name: str = "a.jpg") -> Path:
    img_path = tmp_path / name
    Image.new("RGB", (32, 32), color="red").save(img_path)
    cat.upsert_photo(sha, img_path, filesize=img_path.stat().st_size, mtime=img_path.stat().st_mtime)
    return img_path


def test_runner_predicts_and_stores(catalog: Catalog, tmp_path: Path):
    _seed_photo(catalog, "sha1", tmp_path)
    GeoRunner(catalog=catalog, geolocator=MockGeolocator(top_k=3)).run()

    preds = catalog.get_geo_predictions("sha1", "mock")
    assert len(preds) == 3


def test_runner_skips_already_predicted(catalog: Catalog, tmp_path: Path):
    _seed_photo(catalog, "sha1", tmp_path)
    GeoRunner(catalog=catalog, geolocator=MockGeolocator(top_k=2)).run()

    geo = MockGeolocator(top_k=2)
    calls = {"n": 0}
    real = geo.predict
    geo.predict = lambda imgs: (calls.__setitem__("n", calls["n"] + 1), real(imgs))[1]

    GeoRunner(catalog=catalog, geolocator=geo).run()
    assert calls["n"] == 0


def test_runner_force_repredicts(catalog: Catalog, tmp_path: Path):
    _seed_photo(catalog, "sha1", tmp_path)
    GeoRunner(catalog=catalog, geolocator=MockGeolocator(top_k=2)).run()

    geo = MockGeolocator(top_k=2)
    calls = {"n": 0}
    real = geo.predict
    geo.predict = lambda imgs: (calls.__setitem__("n", calls["n"] + 1), real(imgs))[1]

    GeoRunner(catalog=catalog, geolocator=geo, force=True).run()
    assert calls["n"] >= 1


def test_runner_marks_decode_errors(catalog: Catalog, tmp_path: Path):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not an image")
    catalog.upsert_photo("sha-bad", bad, filesize=bad.stat().st_size, mtime=bad.stat().st_mtime)

    GeoRunner(catalog=catalog, geolocator=MockGeolocator(top_k=2)).run()
    assert catalog.get_photo("sha-bad")["error_reason"] is not None


def test_runner_force_retries_errored_photos(catalog: Catalog, tmp_path: Path):
    _seed_photo(catalog, "sha-x", tmp_path)
    catalog.mark_error("sha-x", "boom")

    GeoRunner(catalog=catalog, geolocator=MockGeolocator(top_k=2)).run()
    assert catalog.get_geo_predictions("sha-x", "mock") == []
    assert catalog.get_photo("sha-x")["error_reason"] is not None

    GeoRunner(catalog=catalog, geolocator=MockGeolocator(top_k=2), force=True).run()
    assert len(catalog.get_geo_predictions("sha-x", "mock")) == 2
    assert catalog.get_photo("sha-x")["error_reason"] is None


def test_runner_replaces_predictions_on_force(catalog: Catalog, tmp_path: Path):
    """Re-running with --force should overwrite stored predictions, not append duplicates."""
    img_path = tmp_path / "a.jpg"
    Image.new("RGB", (32, 32), color="red").save(img_path)
    catalog.upsert_photo("sha1", img_path, filesize=img_path.stat().st_size, mtime=img_path.stat().st_mtime)

    GeoRunner(catalog=catalog, geolocator=MockGeolocator(top_k=3)).run()
    GeoRunner(catalog=catalog, geolocator=MockGeolocator(top_k=3), force=True).run()

    preds = catalog.get_geo_predictions("sha1", "mock")
    assert len(preds) == 3  # not 6


def test_runner_skips_photos_with_camera_gps_by_default(catalog: Catalog, tmp_path: Path):
    _seed_photo(catalog, "sha-no-gps", tmp_path, name="a.jpg")
    img_with_gps = _seed_photo(catalog, "sha-has-gps", tmp_path, name="b.jpg")
    catalog.set_camera_gps("sha-has-gps", latitude=10.0, longitude=20.0, altitude=None)

    GeoRunner(catalog=catalog, geolocator=MockGeolocator(top_k=2)).run()

    assert catalog.get_geo_predictions("sha-no-gps", "mock") != []
    assert catalog.get_geo_predictions("sha-has-gps", "mock") == []


def test_runner_predict_all_includes_photos_with_camera_gps(catalog: Catalog, tmp_path: Path):
    _seed_photo(catalog, "sha-has-gps", tmp_path, name="b.jpg")
    catalog.set_camera_gps("sha-has-gps", latitude=10.0, longitude=20.0, altitude=None)

    GeoRunner(
        catalog=catalog,
        geolocator=MockGeolocator(top_k=2),
        include_with_camera_gps=True,
    ).run()

    assert catalog.get_geo_predictions("sha-has-gps", "mock") != []
