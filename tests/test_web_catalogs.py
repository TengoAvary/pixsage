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


def test_add_catalog_with_valid_path(tmp_path: Path) -> None:
    from pixsage.web.app import build_app
    sony = tmp_path / "Sony"
    _make_catalog(sony / ".photoindex", photo_root=sony)

    registry_path = tmp_path / "catalogs.json"
    app = build_app(registry_path=registry_path, embedder_name="mock")
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
    app = build_app(registry_path=registry_path, embedder_name="mock")
    with TestClient(app) as client:
        r = client.post("/catalogs/add", data={"path": str(bare.resolve())})
        assert r.status_code == 400
        assert ".photoindex" in r.text


def test_add_catalog_with_nonexistent_path(tmp_path: Path) -> None:
    from pixsage.web.app import build_app
    registry_path = tmp_path / "catalogs.json"
    app = build_app(registry_path=registry_path, embedder_name="mock")
    with TestClient(app) as client:
        r = client.post("/catalogs/add", data={"path": "/totally/fake/path"})
        assert r.status_code == 400
        assert "exist" in r.text.lower()


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


def test_rescan_picks_up_new_catalog(tmp_path: Path, monkeypatch) -> None:
    from pixsage.web.app import build_app
    from pixsage import discovery as discovery_mod

    sony = tmp_path / "Sony"
    _make_catalog(sony / ".photoindex", photo_root=sony)

    registry_path = tmp_path / "catalogs.json"
    app = build_app(registry_path=registry_path, embedder_name="mock")

    # Stub list_mounted_roots so rescan sees tmp_path as a root.
    monkeypatch.setattr(discovery_mod, "list_mounted_roots", lambda: [tmp_path])

    with TestClient(app) as client:
        r = client.post("/catalogs/rescan", follow_redirects=False)
        assert r.status_code in (302, 303)
        reg2 = Registry(registry_path)
        reg2.load()
        entries = list(reg2.entries())
        assert len(entries) == 1
        assert entries[0].label == "Sony"


def test_rescan_reloads_after_offline_then_back(tmp_path: Path, monkeypatch) -> None:
    """A catalog that goes offline and comes back should reload into MultiSearchService."""
    from pixsage.web.app import build_app
    from pixsage import discovery as discovery_mod

    sony = tmp_path / "Sony"
    _make_catalog(sony / ".photoindex", photo_root=sony)

    registry_path = tmp_path / "catalogs.json"
    app = build_app(registry_path=registry_path, embedder_name="mock")

    # Step 1: rescan picks up Sony, loads it.
    monkeypatch.setattr(discovery_mod, "list_mounted_roots", lambda: [tmp_path])
    with TestClient(app) as client:
        client.post("/catalogs/rescan")
        assert sony.name in {next(e for e in app.state.registry.entries()).label}
        assert len(app.state.multi_search.catalog_ids()) == 1

        # Step 2: "unplug" — move the .photoindex/ aside so it's offline.
        # Close the loaded catalog's sqlite connection first so Windows lets
        # us rename the directory (POSIX would allow the rename either way).
        # gc.collect() forces release of any lingering sqlite cursor handles.
        import gc
        for cat in list(app.state.catalogs.values()):
            cat.close()
        gc.collect()
        offline_loc = tmp_path / "Sony.photoindex.offline"
        (sony / ".photoindex").rename(offline_loc)
        client.post("/catalogs/rescan")
        assert len(app.state.multi_search.catalog_ids()) == 0

        # Step 3: "replug" — restore the .photoindex/.
        offline_loc.rename(sony / ".photoindex")
        client.post("/catalogs/rescan")
        assert len(app.state.multi_search.catalog_ids()) == 1


def test_add_catalog_with_photoindex_path_directly_uses_parent_label(tmp_path: Path) -> None:
    """User pastes /path/Sony/.photoindex directly — label should be 'Sony', not '.photoindex'."""
    from pixsage.web.app import build_app
    sony = tmp_path / "Sony"
    _make_catalog(sony / ".photoindex", photo_root=sony)

    registry_path = tmp_path / "catalogs.json"
    app = build_app(registry_path=registry_path, embedder_name="mock")
    with TestClient(app) as client:
        photoindex_path = (sony / ".photoindex").resolve()
        r = client.post("/catalogs/add", data={"path": str(photoindex_path)},
                        follow_redirects=False)
        assert r.status_code in (302, 303)
        reg2 = Registry(registry_path)
        reg2.load()
        entries = list(reg2.entries())
        assert len(entries) == 1
        assert entries[0].label == "Sony"  # NOT ".photoindex"


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
