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

    app = build_app(registry_path=registry_path, embedder_name="mock")
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "Sony" in r.text
        assert "iPhone" in r.text


def test_panel_renders_empty_state_when_no_catalogs(tmp_path: Path) -> None:
    from pixsage.web.app import build_app
    registry_path = tmp_path / "catalogs.json"
    app = build_app(registry_path=registry_path, embedder_name="mock")
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

    app = build_app(registry_path=registry_path, embedder_name="mock")
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

    app = build_app(registry_path=registry_path, embedder_name="mock")
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
    app = build_app(registry_path=registry_path, embedder_name="mock")
    with TestClient(app) as client:
        r = client.post("/catalogs/nonexistent/toggle")
        assert r.status_code == 404


def test_remove_deletes_from_registry(tmp_path: Path) -> None:
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

    app = build_app(registry_path=registry_path, embedder_name="mock")
    with TestClient(app) as client:
        r = client.post(f"/catalogs/{e.id}/remove", follow_redirects=False)
        assert r.status_code in (302, 303)
        reg2 = Registry(registry_path)
        reg2.load()
        assert list(reg2.entries()) == []


def test_rename_updates_label(tmp_path: Path) -> None:
    from pixsage.web.app import build_app
    sony = tmp_path / "Sony"
    _make_catalog(sony / ".photoindex", photo_root=sony)

    registry_path = tmp_path / "catalogs.json"
    reg = Registry(registry_path)
    reg.load()
    e = reg.add(photoindex_path=str((sony / ".photoindex").resolve()),
                label="Sony",
                image_embedder_signature="x",
                caption_embedder_signature="y")
    reg.save()

    app = build_app(registry_path=registry_path, embedder_name="mock")
    with TestClient(app) as client:
        r = client.post(f"/catalogs/{e.id}/rename",
                        data={"label": "α7c Sony"}, follow_redirects=False)
        assert r.status_code in (302, 303)
        reg2 = Registry(registry_path)
        reg2.load()
        assert reg2.find_by_id(e.id).label == "α7c Sony"


def test_browse_lists_child_dirs_and_photoindex_hint(tmp_path):
    from pixsage.web.app import build_app
    (tmp_path / "Sony").mkdir()
    (tmp_path / "Sony" / ".photoindex").mkdir()
    (tmp_path / "Empty").mkdir()
    app = build_app(registry_path=tmp_path / "catalogs.json", embedder_name="mock")
    with TestClient(app) as client:
        r = client.get("/catalogs/browse", params={"path": str(tmp_path)})
        assert r.status_code == 200
        body = r.json()
        assert body["path"] == str(tmp_path.resolve())
        names = {e["name"]: e for e in body["entries"]}
        assert names["Sony"]["has_photoindex"] is True
        assert names["Empty"]["has_photoindex"] is False
        assert body["parent"] == str(tmp_path.resolve().parent)


def test_browse_rejects_bad_path(tmp_path):
    from pixsage.web.app import build_app
    app = build_app(registry_path=tmp_path / "catalogs.json", embedder_name="mock")
    with TestClient(app) as client:
        r = client.get("/catalogs/browse", params={"path": str(tmp_path / "nope")})
        assert r.status_code == 400


def test_add_scan_registers_nested_catalogs(tmp_path):
    from pixsage.web.app import build_app
    a = tmp_path / "drive" / "ShootA"
    b = tmp_path / "drive" / "ShootB"
    _make_catalog(a / ".photoindex", photo_root=a)
    _make_catalog(b / ".photoindex", photo_root=b)
    app = build_app(registry_path=tmp_path / "catalogs.json", embedder_name="mock")
    with TestClient(app) as client:
        r = client.post("/catalogs/add-scan",
                         data={"path": str(tmp_path / "drive")},
                         follow_redirects=False)
        assert r.status_code in (302, 303)
        reg = Registry(tmp_path / "catalogs.json")
        reg.load()
        labels = sorted(e.label for e in reg.entries())
        assert labels == ["ShootA", "ShootB"]


def test_add_scan_empty_subtree_adds_nothing(tmp_path):
    from pixsage.web.app import build_app
    (tmp_path / "plain").mkdir()
    app = build_app(registry_path=tmp_path / "catalogs.json", embedder_name="mock")
    with TestClient(app) as client:
        r = client.post("/catalogs/add-scan", data={"path": str(tmp_path / "plain")},
                         follow_redirects=False)
        assert r.status_code in (302, 303)
        reg = Registry(tmp_path / "catalogs.json")
        reg.load()
        assert list(reg.entries()) == []


def test_add_scan_dedupes_already_registered(tmp_path):
    from pixsage.web.app import build_app
    s = tmp_path / "Sony"
    _make_catalog(s / ".photoindex", photo_root=s)
    app = build_app(registry_path=tmp_path / "catalogs.json", embedder_name="mock")
    with TestClient(app) as client:
        client.post("/catalogs/add-scan", data={"path": str(tmp_path)})
        client.post("/catalogs/add-scan", data={"path": str(tmp_path)})
        reg = Registry(tmp_path / "catalogs.json")
        reg.load()
        assert len(list(reg.entries())) == 1


def test_refresh_marks_offline_then_back(tmp_path):
    from pixsage.web.app import build_app
    s = tmp_path / "Sony"
    _make_catalog(s / ".photoindex", photo_root=s)
    app = build_app(registry_path=tmp_path / "catalogs.json", embedder_name="mock")
    with TestClient(app) as client:
        client.post("/catalogs/add-scan", data={"path": str(tmp_path)})
        assert len(app.state.multi_search.catalog_ids()) == 1

        import gc
        for cat in list(app.state.catalogs.values()):
            cat.close()
        gc.collect()
        offline = tmp_path / "Sony.photoindex.offline"
        (s / ".photoindex").rename(offline)
        client.post("/catalogs/refresh")
        assert len(app.state.multi_search.catalog_ids()) == 0

        offline.rename(s / ".photoindex")
        client.post("/catalogs/refresh")
        assert len(app.state.multi_search.catalog_ids()) == 1


def test_add_scan_empty_reports_notice(tmp_path):
    from pixsage.web.app import build_app
    (tmp_path / "plain").mkdir()
    app = build_app(registry_path=tmp_path / "catalogs.json", embedder_name="mock")
    with TestClient(app) as client:
        r = client.post("/catalogs/add-scan", data={"path": str(tmp_path / "plain")},
                         follow_redirects=False)
        assert r.status_code == 303
        loc = r.headers["location"]
        assert "notice=" in loc
        assert "nothing%20added" in loc or "nothing+added" in loc


def test_add_scan_success_reports_count_and_renders_notice(tmp_path):
    from pixsage.web.app import build_app
    s = tmp_path / "Sony"
    _make_catalog(s / ".photoindex", photo_root=s)
    app = build_app(registry_path=tmp_path / "catalogs.json", embedder_name="mock")
    with TestClient(app) as client:
        r = client.post("/catalogs/add-scan", data={"path": str(tmp_path)},
                         follow_redirects=False)
        assert r.status_code == 303
        assert "Added%201" in r.headers["location"]
        # notice renders on the page
        r2 = client.get("/", params={"notice": "Added 1 catalog(s)"})
        assert "Added 1 catalog(s)" in r2.text
        assert 'class="catalog-notice"' in r2.text


def test_index_has_folder_browser_modal(tmp_path):
    from pixsage.web.app import build_app
    app = build_app(registry_path=tmp_path / "catalogs.json", embedder_name="mock")
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert 'id="catalog-browser"' in r.text
        assert "/catalogs/add-scan" in r.text
        assert "/catalogs/refresh" in r.text
        assert "/catalogs/rescan" not in r.text
        assert "/catalogs/add\"" not in r.text


def test_add_scan_rejects_bad_path(tmp_path):
    from pixsage.web.app import build_app
    app = build_app(registry_path=tmp_path / "catalogs.json", embedder_name="mock")
    with TestClient(app) as client:
        r = client.post("/catalogs/add-scan", data={"path": str(tmp_path / "nope")},
                         follow_redirects=False)
        assert r.status_code == 400


def test_home_renders_catalogs_modal_dialog(tmp_path: Path) -> None:
    """The catalog management UI must be rendered as a `<dialog>` with
    id `catalogs-modal`, not the old `<details>` panel. The inner
    form actions (rename / toggle / remove / refresh / add-scan) and
    the nested folder-picker `<dialog id="catalog-browser">` must
    still be present.
    """
    from pixsage.web.app import build_app

    sony = tmp_path / "Sony"
    _make_catalog(sony / ".photoindex", photo_root=sony)

    registry_path = tmp_path / "catalogs.json"
    reg = Registry(registry_path)
    reg.load()
    reg.add(photoindex_path=str((sony / ".photoindex").resolve()),
            label="Sony",
            image_embedder_signature="siglip2-so400m-patch14-384@v1",
            caption_embedder_signature="minilm-L6-v2@v2")
    reg.save()

    app = build_app(registry_path=registry_path, embedder_name="mock")
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert 'class="catalogs-modal"' in r.text
        assert 'id="catalogs-modal"' in r.text
        # Old wrapper must be gone.
        assert 'class="catalogs-panel"' not in r.text
        # Form actions still present.
        assert "/catalogs/refresh" in r.text
        assert "/catalogs/add-scan" in r.text
        # Folder-picker dialog still present.
        assert 'id="catalog-browser"' in r.text


def test_home_search_slider_lives_in_weight_row(tmp_path: Path) -> None:
    """The Caption ⇄ Visual slider sits in a sibling `.weight` row
    below the search input, not inline with the input.
    """
    from pixsage.web.app import build_app

    registry_path = tmp_path / "catalogs.json"
    app = build_app(registry_path=registry_path, embedder_name="mock")
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert 'class="weight"' in r.text
        # The slider's `name` attribute is unchanged.
        assert 'name="image_weight"' in r.text


def test_home_renders_collapsed_catalogs_strip(tmp_path: Path) -> None:
    """The home page renders a one-line `.catalogs-strip` summary
    above the search form. The full management UI lives in a modal
    that is opened from this strip's `Manage ▸` button.
    """
    from pixsage.web.app import build_app

    sony = tmp_path / "Sony"
    _make_catalog(sony / ".photoindex", photo_root=sony)

    registry_path = tmp_path / "catalogs.json"
    reg = Registry(registry_path)
    reg.load()
    reg.add(photoindex_path=str((sony / ".photoindex").resolve()),
            label="Sony",
            image_embedder_signature="siglip2-so400m-patch14-384@v1",
            caption_embedder_signature="minilm-L6-v2@v2")
    reg.save()

    app = build_app(registry_path=registry_path, embedder_name="mock")
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert 'class="catalogs-strip"' in r.text
        # The strip surfaces a Manage affordance via its own button class.
        assert 'class="cs-manage"' in r.text
