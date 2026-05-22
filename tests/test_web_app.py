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
    )
    with TestClient(app) as client:
        r = client.get("/static/htmx.min.js")
        assert r.status_code == 200
        assert "htmx" in r.text.lower()


def test_startup_does_not_walk_filesystem(tmp_path, monkeypatch):
    """build_app() must not trigger a recursive discovery walk at startup."""
    import pixsage.discovery as disc
    from pixsage.web.app import build_app

    called = {"n": 0}
    monkeypatch.setattr(disc, "walk_for_photoindex",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1) or [])
    build_app(registry_path=tmp_path / "catalogs.json", embedder_name="mock")
    assert called["n"] == 0


def test_deferred_load_eventually_becomes_ready(tmp_path: Path):
    """defer_load=True returns immediately in 'loading', then a background
    thread flips to 'ready' (mock embedder loads instantly)."""
    import time

    from pixsage.web.app import build_app

    app = build_app(
        registry_path=tmp_path / "catalogs.json",
        embedder_name="mock",
        defer_load=True,
    )
    assert app.state.loader.status in ("loading", "ready")

    deadline = time.time() + 5
    while time.time() < deadline and app.state.loader.status != "ready":
        time.sleep(0.05)
    assert app.state.loader.status == "ready"


def _ready_app(tmp_path: Path):
    from pixsage.web.app import build_app

    return build_app(
        registry_path=tmp_path / "catalogs.json",
        embedder_name="mock",
        defer_load=False,
    )


def test_status_endpoint_reports_ready(tmp_path: Path):
    app = _ready_app(tmp_path)
    with TestClient(app) as client:
        body = client.get("/status").json()
        assert body["status"] == "ready"
        assert all(p["state"] == "done" for p in body["phases"])
        assert body["error"] is None


def test_index_shows_loading_screen_when_not_ready(tmp_path: Path):
    app = _ready_app(tmp_path)
    app.state.loader.status = "loading"  # drive loading state deterministically
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "warming up" in r.text.lower() or "loading" in r.text.lower()
        assert client.get("/status").json()["status"] == "loading"


def test_gated_route_returns_503_while_loading_but_status_and_static_ok(tmp_path: Path):
    app = _ready_app(tmp_path)
    app.state.loader.status = "loading"
    with TestClient(app) as client:
        assert client.get("/thumb/cat/sha").status_code == 503
        assert client.get("/status").status_code == 200
        assert client.get("/static/htmx.min.js").status_code == 200


def test_index_shows_search_page_when_ready(tmp_path: Path):
    app = _ready_app(tmp_path)
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "search" in r.text.lower()


def test_loading_page_has_poller_and_phase_markup(tmp_path: Path):
    app = _ready_app(tmp_path)
    app.state.loader.status = "loading"
    with TestClient(app) as client:
        html = client.get("/").text.lower()
        assert "/status" in html          # JS polls the status endpoint
        assert "pixsage" in html
        assert "phases" in html            # renders the phase checklist
