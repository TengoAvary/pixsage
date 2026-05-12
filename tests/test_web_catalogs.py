from __future__ import annotations
from pathlib import Path

from fastapi.testclient import TestClient

from pixsage.catalog import Catalog
from pixsage.registry import Registry


def _make_catalog(photoindex: Path, *, photo_root: Path) -> None:
    photoindex.mkdir(parents=True, exist_ok=True)
    cat = Catalog(photoindex / "catalog.db")
    cat.init_schema()
    cat.set_photo_root_if_unset(photo_root)
    photo_root.mkdir(parents=True, exist_ok=True)


def test_panel_renders_two_catalogs(tmp_path: Path) -> None:
    from pixsage.web.app import build_app
    sony = tmp_path / "Sony"
    iphone = tmp_path / "iPhone"
    _make_catalog(sony / ".photoindex", photo_root=sony)
    _make_catalog(iphone / ".photoindex", photo_root=iphone)

    registry_path = tmp_path / "catalogs.json"
    reg = Registry(registry_path)
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

    app = build_app(registry_path=registry_path, embedder_name="mock", skip_discovery=True)
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "Sony" in r.text
        assert "iPhone" in r.text


def test_panel_renders_empty_state_when_no_catalogs(tmp_path: Path) -> None:
    from pixsage.web.app import build_app
    registry_path = tmp_path / "catalogs.json"
    app = build_app(registry_path=registry_path, embedder_name="mock", skip_discovery=True)
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "No catalogs" in r.text


def test_panel_shows_offline_for_unreachable_path(tmp_path: Path) -> None:
    from pixsage.web.app import build_app
    registry_path = tmp_path / "catalogs.json"
    reg = Registry(registry_path)
    reg.load()
    reg.add(photoindex_path="/Volumes/NotMounted/.photoindex",
            label="Offline Drive",
            image_embedder_signature="x",
            caption_embedder_signature="y")
    reg.save()

    app = build_app(registry_path=registry_path, embedder_name="mock", skip_discovery=True)
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "Offline Drive" in r.text
        assert "offline" in r.text.lower()


def test_toggle_disables_catalog(tmp_path: Path) -> None:
    from pixsage.web.app import build_app
    sony = tmp_path / "Sony"
    _make_catalog(sony / ".photoindex", photo_root=sony)

    registry_path = tmp_path / "catalogs.json"
    reg = Registry(registry_path)
    reg.load()
    e = reg.add(photoindex_path=str((sony / ".photoindex").resolve()),
                label="Sony",
                image_embedder_signature="siglip2-so400m-patch14-384@v1",
                caption_embedder_signature="minilm-L6-v2@v2")
    reg.save()

    app = build_app(registry_path=registry_path, embedder_name="mock", skip_discovery=True)
    with TestClient(app) as client:
        r = client.post(f"/catalogs/{e.id}/toggle", follow_redirects=False)
        assert r.status_code in (302, 303)
        # Reload registry from disk to confirm persisted
        reg2 = Registry(registry_path)
        reg2.load()
        assert reg2.find_by_id(e.id).enabled is False


def test_toggle_unknown_id_returns_404(tmp_path: Path) -> None:
    from pixsage.web.app import build_app
    registry_path = tmp_path / "catalogs.json"
    app = build_app(registry_path=registry_path, embedder_name="mock", skip_discovery=True)
    with TestClient(app) as client:
        r = client.post("/catalogs/nonexistent/toggle")
        assert r.status_code == 404


def test_add_catalog_with_valid_path(tmp_path: Path) -> None:
    from pixsage.web.app import build_app
    sony = tmp_path / "Sony"
    _make_catalog(sony / ".photoindex", photo_root=sony)

    registry_path = tmp_path / "catalogs.json"
    app = build_app(registry_path=registry_path, embedder_name="mock", skip_discovery=True)
    with TestClient(app) as client:
        r = client.post(
            "/catalogs/add",
            data={"path": str(sony.resolve())},
            follow_redirects=False,
        )
        assert r.status_code in (302, 303)
        reg2 = Registry(registry_path)
        reg2.load()
        entries = list(reg2.entries())
        assert len(entries) == 1
        assert entries[0].label == "Sony"


def test_add_catalog_with_missing_photoindex(tmp_path: Path) -> None:
    from pixsage.web.app import build_app
    bare = tmp_path / "NoCatalogHere"
    bare.mkdir()

    registry_path = tmp_path / "catalogs.json"
    app = build_app(registry_path=registry_path, embedder_name="mock", skip_discovery=True)
    with TestClient(app) as client:
        r = client.post("/catalogs/add", data={"path": str(bare.resolve())})
        assert r.status_code == 400
        assert ".photoindex" in r.text


def test_add_catalog_with_nonexistent_path(tmp_path: Path) -> None:
    from pixsage.web.app import build_app
    registry_path = tmp_path / "catalogs.json"
    app = build_app(registry_path=registry_path, embedder_name="mock", skip_discovery=True)
    with TestClient(app) as client:
        r = client.post("/catalogs/add", data={"path": "/totally/fake/path"})
        assert r.status_code == 400
        assert "exist" in r.text.lower()
