from pathlib import Path

import pytest


def test_app_state_has_path_resolver(tmp_path: Path) -> None:
    """build_app constructs a PathResolver from the catalog meta and
    runtime photo_root, exposed on app.state.path_resolver."""
    from pixsage.catalog import Catalog
    from pixsage.web.app import build_app

    photo_root = tmp_path / "drive" / "Sony alpha 7c"
    photo_root.mkdir(parents=True)

    cat_path = photo_root / ".photoindex" / "catalog.db"
    cat_path.parent.mkdir(parents=True, exist_ok=True)
    cat = Catalog(cat_path)
    cat.init_schema()
    cat.set_photo_root_if_unset(Path(r"E:\Sony alpha 7c"))
    cat.close()

    app = build_app(photo_root=photo_root, embedder_name="mock")
    resolver = app.state.path_resolver
    # Translation: stored prefix E:\Sony alpha 7c → runtime tmp_path/drive/Sony alpha 7c
    target = photo_root / "DSC_1234.ARW"
    target.write_bytes(b"raw")
    resolved = resolver.resolve(r"E:\Sony alpha 7c\DSC_1234.ARW")
    assert resolved == target


import io

from fastapi.testclient import TestClient
from PIL import Image


def _make_jpeg(path: Path, color: str = "red") -> None:
    img = Image.new("RGB", (32, 32), color=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG")


def test_thumb_route_resolves_translated_path(tmp_path: Path) -> None:
    """Catalog has a current_path of E:\\Sony alpha 7c\\DSC_0001.JPG,
    file actually lives at tmp_path/drive/Sony alpha 7c/DSC_0001.JPG.
    /grid/thumb/<sha> should serve it."""
    from pixsage.catalog import Catalog
    from pixsage.web.app import build_app

    photo_root = tmp_path / "drive" / "Sony alpha 7c"
    photo_root.mkdir(parents=True)
    real = photo_root / "DSC_0001.JPG"
    _make_jpeg(real)

    cat_path = photo_root / ".photoindex" / "catalog.db"
    cat_path.parent.mkdir(parents=True, exist_ok=True)
    cat = Catalog(cat_path)
    cat.init_schema()
    cat.set_photo_root_if_unset(Path(r"E:\Sony alpha 7c"))
    # Insert photo with the FAKE Windows path; then the resolver has work to do.
    cat._conn.execute(
        "INSERT INTO photos (sha256, current_path, filename, filesize, mtime, added_at, last_seen_at) "
        "VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'))",
        ("abc123", r"E:\Sony alpha 7c\DSC_0001.JPG", "DSC_0001.JPG", real.stat().st_size, real.stat().st_mtime),
    )
    cat._conn.commit()
    cat.close()

    app = build_app(photo_root=photo_root, embedder_name="mock")
    client = TestClient(app)
    r = client.get("/thumb/abc123?size=small")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("image/")
