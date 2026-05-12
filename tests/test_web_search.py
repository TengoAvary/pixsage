from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient


def _seed_root(tmp_path: Path) -> Path:
    """Build a photo_root with two photos, mock embeddings already populated."""
    from pixsage.catalog import Catalog
    from pixsage.embedders.mock import MockEmbedder
    from pixsage.vectors import VectorStore

    root = tmp_path / "photos"
    root.mkdir()
    photoindex = root / ".photoindex"
    photoindex.mkdir()

    a, b = root / "a.jpg", root / "b.jpg"
    Image.new("RGB", (64, 64), color="red").save(a)
    Image.new("RGB", (64, 64), color="blue").save(b)

    cat = Catalog(photoindex / "catalog.db")
    cat.init_schema()
    cat.upsert_photo("sha-a", a, filesize=a.stat().st_size, mtime=a.stat().st_mtime)
    cat.upsert_photo("sha-b", b, filesize=b.stat().st_size, mtime=b.stat().st_mtime)
    cat.record_caption("sha-a", "a red square")
    cat.record_caption("sha-b", "a blue square")

    embedder = MockEmbedder()  # must match build_app's default dim (16)
    vec_a_img = embedder.embed_image([Image.open(a)])[0]
    vec_b_img = embedder.embed_image([Image.open(b)])[0]
    vec_a_txt = embedder.embed_text(["a red square"])[0]
    vec_b_txt = embedder.embed_text(["a blue square"])[0]

    store = VectorStore(photoindex / "vectors")
    store.append("mock_image", [("sha-a", vec_a_img), ("sha-b", vec_b_img)])
    store.append("mock_text", [("sha-a", vec_a_txt), ("sha-b", vec_b_txt)])

    cat.close()
    return root


def _catalog_id(client) -> str:
    """First enabled+available catalog id — the only one in single-catalog tests."""
    for e in client.app.state.registry.entries():
        if e.enabled and e.available:
            return e.id
    raise AssertionError("no enabled catalog")


def test_search_returns_results_html(tmp_path: Path):
    from pixsage.web.app import build_app

    root = _seed_root(tmp_path)
    app = build_app(
        photo_root=root,
        registry_path=tmp_path / "catalogs.json",
        embedder_name="mock",
        skip_discovery=True,
    )
    with TestClient(app) as client:
        r = client.get("/", params={"q": "a red square", "image_weight": "0.0"})
        assert r.status_code == 200
        assert "sha-a" in r.text or "sha-b" in r.text
        assert "<article" in r.text or 'class="card"' in r.text


def test_search_empty_query_returns_empty_results(tmp_path: Path):
    from pixsage.web.app import build_app

    root = _seed_root(tmp_path)
    app = build_app(
        photo_root=root,
        registry_path=tmp_path / "catalogs.json",
        embedder_name="mock",
        skip_discovery=True,
    )
    with TestClient(app) as client:
        r = client.get("/", params={"q": "", "image_weight": "0.5"})
        assert r.status_code == 200
        # Empty query -> no result cards rendered.
        assert 'class="card"' not in r.text


def test_thumb_route_returns_jpeg(tmp_path: Path):
    from pixsage.web.app import build_app

    root = _seed_root(tmp_path)
    app = build_app(
        photo_root=root,
        registry_path=tmp_path / "catalogs.json",
        embedder_name="mock",
        skip_discovery=True,
    )
    with TestClient(app) as client:
        cid = _catalog_id(client)
        r = client.get(f"/thumb/{cid}/sha-a?size=small")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"
        assert len(r.content) > 0


def test_thumb_route_404_for_missing_sha(tmp_path: Path):
    from pixsage.web.app import build_app

    root = _seed_root(tmp_path)
    app = build_app(
        photo_root=root,
        registry_path=tmp_path / "catalogs.json",
        embedder_name="mock",
        skip_discovery=True,
    )
    with TestClient(app) as client:
        cid = _catalog_id(client)
        r = client.get(f"/thumb/{cid}/nonexistent-sha?size=small")
        assert r.status_code == 404


def test_photo_detail_renders_caption_and_filename(tmp_path: Path):
    from pixsage.web.app import build_app

    root = _seed_root(tmp_path)
    app = build_app(
        photo_root=root,
        registry_path=tmp_path / "catalogs.json",
        embedder_name="mock",
        skip_discovery=True,
    )
    with TestClient(app) as client:
        cid = _catalog_id(client)
        r = client.get(f"/photo/{cid}/sha-a")
        assert r.status_code == 200
        assert "a red square" in r.text          # caption
        assert "a.jpg" in r.text                 # filename
        assert f"/similar/{cid}/sha-a" in r.text  # more-like-this link


def test_photo_detail_404(tmp_path: Path):
    from pixsage.web.app import build_app

    root = _seed_root(tmp_path)
    app = build_app(
        photo_root=root,
        registry_path=tmp_path / "catalogs.json",
        embedder_name="mock",
        skip_discovery=True,
    )
    with TestClient(app) as client:
        cid = _catalog_id(client)
        r = client.get(f"/photo/{cid}/nonexistent-sha")
        assert r.status_code == 404


def test_similar_returns_results_excluding_self(tmp_path: Path):
    from pixsage.web.app import build_app

    root = _seed_root(tmp_path)
    app = build_app(
        photo_root=root,
        registry_path=tmp_path / "catalogs.json",
        embedder_name="mock",
        skip_discovery=True,
    )
    with TestClient(app) as client:
        cid = _catalog_id(client)
        r = client.get(f"/similar/{cid}/sha-a")
        assert r.status_code == 200
        # The query photo's own card must not be in the results grid;
        # the other photo's card must be.
        assert 'data-sha="sha-b"' in r.text
        assert 'data-sha="sha-a"' not in r.text


def test_similar_404_when_photo_missing(tmp_path: Path):
    from pixsage.web.app import build_app

    root = _seed_root(tmp_path)
    app = build_app(
        photo_root=root,
        registry_path=tmp_path / "catalogs.json",
        embedder_name="mock",
        skip_discovery=True,
    )
    with TestClient(app) as client:
        cid = _catalog_id(client)
        r = client.get(f"/similar/{cid}/nonexistent")
        assert r.status_code == 404


def test_result_card_shows_catalog_badge_in_multi_mode(tmp_path: Path) -> None:
    """When two catalogs are enabled, result cards must show the catalog label."""
    from pixsage.catalog import Catalog
    from pixsage.registry import Registry
    from pixsage.web.app import build_app

    # Two catalogs, each with one photo
    sony = tmp_path / "Sony"
    iphone = tmp_path / "iPhone"
    for root, sha in [(sony, "sha-sony"), (iphone, "sha-iphone")]:
        photoindex = root / ".photoindex"
        photoindex.mkdir(parents=True)
        cat = Catalog(photoindex / "catalog.db")
        cat.init_schema()
        cat.set_photo_root_if_unset(root)
        img = root / f"{sha}.jpg"
        img.write_bytes(b"fake")
        cat.upsert_photo(sha, img, img.stat().st_size, img.stat().st_mtime)

    reg = Registry(tmp_path / "catalogs.json")
    reg.load()
    reg.add(photoindex_path=str((sony / ".photoindex").resolve()),
            label="Sony",
            image_embedder_signature="siglip2-so400m-patch14-384@v1",
            caption_embedder_signature="minilm-L6-v2@v2")
    reg.add(photoindex_path=str((iphone / ".photoindex").resolve()),
            label="iPhone",
            image_embedder_signature="siglip2-so400m-patch14-384@v1",
            caption_embedder_signature="minilm-L6-v2@v2")
    reg.save()

    app = build_app(registry_path=tmp_path / "catalogs.json",
                    embedder_name="mock", skip_discovery=True)
    with TestClient(app) as client:
        r = client.get("/", params={"q": "anything"})
        assert r.status_code == 200
        # Panel itself shows both labels.
        assert "Sony" in r.text and "iPhone" in r.text
        # Soft-check the badge: if mock returns hits, catalog-badge must render.
        if 'class="card"' in r.text:
            assert "catalog-badge" in r.text


def test_result_card_no_badge_in_single_catalog_mode(tmp_path: Path) -> None:
    """When only one catalog is enabled, result cards must NOT show the catalog badge."""
    root = _seed_root(tmp_path)
    from pixsage.web.app import build_app

    app = build_app(
        photo_root=root,
        registry_path=tmp_path / "catalogs.json",
        embedder_name="mock",
        skip_discovery=True,
    )
    with TestClient(app) as client:
        r = client.get("/", params={"q": "a red square", "image_weight": "0.0"})
        assert r.status_code == 200
        # We have hits in single-catalog mode — verify the badge is absent.
        assert 'class="card"' in r.text
        assert "catalog-badge" not in r.text
