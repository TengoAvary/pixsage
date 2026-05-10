from __future__ import annotations

from PIL import Image

from pixsage.geolocators.base import GeoPrediction
from pixsage.geolocators.mock import MockGeolocator


def test_mock_returns_top_k_predictions():
    geo = MockGeolocator(top_k=3)
    img = Image.new("RGB", (16, 16), color="red")
    out = geo.predict([img])
    assert len(out) == 1
    assert len(out[0]) == 3
    for p in out[0]:
        assert isinstance(p, GeoPrediction)
        assert -90.0 <= p.latitude <= 90.0
        assert -180.0 <= p.longitude <= 180.0
        assert 0.0 < p.score <= 1.0


def test_mock_predictions_descending_score():
    geo = MockGeolocator(top_k=4)
    img = Image.new("RGB", (16, 16), color="red")
    preds = geo.predict([img])[0]
    scores = [p.score for p in preds]
    assert scores == sorted(scores, reverse=True)


def test_mock_deterministic_for_same_image():
    geo = MockGeolocator(top_k=3)
    img = Image.new("RGB", (16, 16), color="blue")
    p1 = geo.predict([img])[0]
    p2 = geo.predict([img])[0]
    coords_1 = [(p.latitude, p.longitude, p.score) for p in p1]
    coords_2 = [(p.latitude, p.longitude, p.score) for p in p2]
    assert coords_1 == coords_2


def test_mock_distinguishes_different_images():
    geo = MockGeolocator(top_k=3)
    img_a = Image.new("RGB", (16, 16), color="red")
    img_b = Image.new("RGB", (16, 16), color="blue")
    pa = geo.predict([img_a])[0]
    pb = geo.predict([img_b])[0]
    coords_a = [(p.latitude, p.longitude) for p in pa]
    coords_b = [(p.latitude, p.longitude) for p in pb]
    assert coords_a != coords_b


def test_mock_handles_batch():
    geo = MockGeolocator(top_k=2)
    imgs = [Image.new("RGB", (8, 8), color=c) for c in ("red", "green", "blue")]
    out = geo.predict(imgs)
    assert len(out) == 3
    for preds in out:
        assert len(preds) == 2


def test_mock_info_metadata():
    geo = MockGeolocator(top_k=7)
    assert geo.info.name == "mock"
    assert geo.info.top_k == 7
