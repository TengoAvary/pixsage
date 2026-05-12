from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient

from pixsage.catalog import Catalog


def _make_catalog(photoindex: Path, *, photo_root: Path) -> None:
    photoindex.mkdir(parents=True, exist_ok=True)
    cat = Catalog(photoindex / "catalog.db")
    cat.init_schema()
    cat.set_photo_root_if_unset(photo_root)
    photo_root.mkdir(parents=True, exist_ok=True)
    # Insert one fake photo so panel render has something to count
    img = photo_root / "a.jpg"
    img.write_bytes(b"fake")
    cat.upsert_photo("sha-a", img, img.stat().st_size, img.stat().st_mtime)
    cat.close()


def test_build_app_with_empty_registry_serves_empty_state(tmp_path: Path) -> None:
    """When the registry is empty and no photo_root is given, the app starts
    and renders the empty-state catalog panel."""
    from pixsage.web.app import build_app
    registry_path = tmp_path / "catalogs.json"
    app = build_app(
        registry_path=registry_path,
        embedder_name="mock",
        skip_discovery=True,
    )
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        # TODO(Task 10): re-enable content assertion after catalog panel ships.
        # assert "No catalogs" in r.text or "add a catalog" in r.text.lower()


def test_build_app_with_single_photo_root_auto_registers(tmp_path: Path) -> None:
    """Backward-compat: build_app(photo_root=...) adds that path to the registry."""
    from pixsage.web.app import build_app
    photo_root = tmp_path / "Sony"
    _make_catalog(photo_root / ".photoindex", photo_root=photo_root)

    registry_path = tmp_path / "catalogs.json"
    app = build_app(
        registry_path=registry_path,
        photo_root=photo_root,
        embedder_name="mock",
        skip_discovery=True,
    )
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        # TODO(Task 10): re-enable content assertion after catalog panel ships.
        # assert "Sony" in r.text

    # Verify it was actually written to the registry.
    from pixsage.registry import Registry
    reg = Registry(registry_path)
    reg.load()
    entries = list(reg.entries())
    assert len(entries) == 1
    assert entries[0].label == "Sony"


def test_build_app_loads_two_catalogs_from_registry(tmp_path: Path) -> None:
    """Two pre-registered catalogs: both loaded."""
    from pixsage.web.app import build_app
    from pixsage.registry import Registry

    sony = tmp_path / "Sony"
    iphone = tmp_path / "iPhone"
    _make_catalog(sony / ".photoindex", photo_root=sony)
    _make_catalog(iphone / ".photoindex", photo_root=iphone)

    registry_path = tmp_path / "catalogs.json"
    reg = Registry(registry_path)
    reg.load()
    reg.add(
        photoindex_path=str((sony / ".photoindex").resolve()),
        label="Sony",
        image_embedder_signature="siglip2-so400m-patch14-384@v1",
        caption_embedder_signature="minilm-L6-v2@v2",
    )
    reg.add(
        photoindex_path=str((iphone / ".photoindex").resolve()),
        label="iPhone",
        image_embedder_signature="siglip2-so400m-patch14-384@v1",
        caption_embedder_signature="minilm-L6-v2@v2",
    )
    reg.save()

    app = build_app(
        registry_path=registry_path,
        embedder_name="mock",
        skip_discovery=True,
    )
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        # TODO(Task 10): re-enable content assertions after catalog panel ships.
        # assert "Sony" in r.text
        # assert "iPhone" in r.text

    # Verify both catalogs were loaded into the multi_search.
    assert len(app.state.catalogs) == 2
    assert len(app.state.multi_search.catalog_ids()) == 2
