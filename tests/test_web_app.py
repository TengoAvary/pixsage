from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient


def test_index_returns_search_page(tmp_path: Path):
    from pixsage.web.app import build_app

    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    (photo_root / ".photoindex").mkdir()

    app = build_app(
        photo_root=photo_root,
        registry_path=tmp_path / "catalogs.json",
        embedder_name="mock",
        skip_discovery=True,
    )
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "pixsage" in r.text.lower()
        assert "search" in r.text.lower()


def test_static_assets_served(tmp_path: Path):
    from pixsage.web.app import build_app

    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    (photo_root / ".photoindex").mkdir()

    app = build_app(
        photo_root=photo_root,
        registry_path=tmp_path / "catalogs.json",
        embedder_name="mock",
        skip_discovery=True,
    )
    with TestClient(app) as client:
        r = client.get("/static/htmx.min.js")
        assert r.status_code == 200
        assert "htmx" in r.text.lower()
