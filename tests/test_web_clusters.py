"""Tests for the experimental HITL cluster labelling routes.

These routes are off by default; build_app(experimental_cluster_labelling=True)
is the opt-in. Tests use a fake cluster list (real UMAP+HDBSCAN compute is
unit-tested elsewhere and too slow for a tight loop).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from PIL import Image
from fastapi.testclient import TestClient

from pixsage.catalog import Catalog
from pixsage.clusters import Cluster
from pixsage.web.app import build_app


def _make_corpus(tmp_path: Path) -> Path:
    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    for name, color in (("a.jpg", "red"), ("b.jpg", "blue"), ("c.jpg", "green")):
        Image.new("RGB", (32, 32), color=color).save(photo_root / name)

    cat_path = photo_root / ".photoindex" / "catalog.db"
    cat_path.parent.mkdir(parents=True, exist_ok=True)
    cat = Catalog(cat_path)
    cat.init_schema()
    from pixsage.walker import sha256_file, walk_photos
    for p in walk_photos(photo_root):
        cat.upsert_photo(sha256_file(p), p, p.stat().st_size, p.stat().st_mtime)
    cat.close()
    return photo_root


def _build_app_with_fake_clusters(photo_root: Path):
    # Isolate per-test registry + skip discovery so the global registry path
    # doesn't leak state between tests.
    registry_path = photo_root.parent / "catalogs.json"
    app = build_app(
        photo_root=photo_root,
        registry_path=registry_path,
        embedder_name="mock",
        experimental_cluster_labelling=True,
        skip_discovery=True,
    )
    cat = next(iter(app.state.catalogs.values()))
    shas = [r["sha256"] for r in cat._conn.execute(  # noqa: SLF001
        "SELECT sha256 FROM photos ORDER BY filename"
    )]
    fake_cluster = Cluster(
        cluster_id=42,
        member_shas=shas,
        sample_shas=shas[:4],
        folder_distribution=[("photos", len(shas))],
        distinctive_tags=["test"],
    )
    app.state.clusters = [fake_cluster]
    return app, shas


def test_explore_renders(tmp_path: Path):
    photo_root = _make_corpus(tmp_path)
    app, _ = _build_app_with_fake_clusters(photo_root)
    client = TestClient(app)
    r = client.get("/explore")
    assert r.status_code == 200
    assert "1 visual clusters" in r.text or "1 visual clusters" in r.text
    assert "cluster/42" in r.text


def test_cluster_detail_renders(tmp_path: Path):
    photo_root = _make_corpus(tmp_path)
    app, _ = _build_app_with_fake_clusters(photo_root)
    client = TestClient(app)
    r = client.get("/cluster/42")
    assert r.status_code == 200
    assert "Apply to all 3 photos" in r.text


def test_cluster_404(tmp_path: Path):
    photo_root = _make_corpus(tmp_path)
    app, _ = _build_app_with_fake_clusters(photo_root)
    client = TestClient(app)
    assert client.get("/cluster/999").status_code == 404


def test_label_apply_writes_catalog_and_xmp(tmp_path: Path):
    photo_root = _make_corpus(tmp_path)
    app, shas = _build_app_with_fake_clusters(photo_root)
    client = TestClient(app, follow_redirects=False)

    write_calls: list[tuple] = []
    def fake_write_gps(path, latitude, longitude, place_name, is_raw):
        write_calls.append((str(path), latitude, longitude, place_name, is_raw))

    with patch("pixsage.xmp.write_gps", fake_write_gps):
        r = client.post(
            "/cluster/42/label",
            data={
                "latitude": -64.2799,
                "longitude": -56.7449,
                "place_name": "Seymour Island",
            },
        )
    assert r.status_code == 303
    assert r.headers["location"] == "/cluster/42"

    # Every photo got a write_gps call
    assert len(write_calls) == len(shas)
    for path, lat, lon, place, is_raw in write_calls:
        assert lat == -64.2799
        assert lon == -56.7449
        assert place == "Seymour Island"
        assert is_raw is False  # JPGs

    # And every photo has a catalog row
    cat = next(iter(app.state.catalogs.values()))
    for sha in shas:
        loc = cat.get_user_location(sha)
        assert loc is not None
        assert loc["latitude"] == -64.2799
        assert loc["place_name"] == "Seymour Island"
        assert loc["applied_via"] == "cluster:42"


def test_cluster_routes_404_when_flag_off(tmp_path: Path):
    """The default `pixsage serve` invocation must not expose these routes."""
    photo_root = _make_corpus(tmp_path)
    app = build_app(
        photo_root=photo_root,
        registry_path=tmp_path / "catalogs.json",
        embedder_name="mock",
        skip_discovery=True,
    )  # flag off by default
    client = TestClient(app)
    assert client.get("/explore").status_code == 404
    assert client.get("/cluster/42").status_code == 404
    r = client.post("/cluster/42/label", data={"latitude": 0, "longitude": 0})
    assert r.status_code == 404


def test_label_apply_with_no_place_name(tmp_path: Path):
    photo_root = _make_corpus(tmp_path)
    app, shas = _build_app_with_fake_clusters(photo_root)
    client = TestClient(app, follow_redirects=False)

    with patch("pixsage.xmp.write_gps"):
        client.post(
            "/cluster/42/label",
            data={"latitude": 0.0, "longitude": 0.0, "place_name": ""},
        )

    cat = next(iter(app.state.catalogs.values()))
    for sha in shas:
        loc = cat.get_user_location(sha)
        assert loc is not None
        assert loc["place_name"] is None
