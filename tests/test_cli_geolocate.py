from __future__ import annotations

import zipfile
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from pixsage.catalog import Catalog
from pixsage.cli import app


runner = CliRunner()


def _make_photo_root(tmp_path: Path) -> Path:
    root = tmp_path / "photos"
    root.mkdir()
    Image.new("RGB", (64, 64), color="red").save(root / "a.jpg")
    Image.new("RGB", (64, 64), color="blue").save(root / "b.jpg")
    return root


def _seed_catalog(photo_root: Path) -> Path:
    cat_path = photo_root / ".photoindex" / "catalog.db"
    cat_path.parent.mkdir(parents=True, exist_ok=True)
    cat = Catalog(cat_path)
    cat.init_schema()
    from pixsage.walker import sha256_file, walk_photos
    for p in walk_photos(photo_root):
        sha = sha256_file(p)
        cat.upsert_photo(sha, p, filesize=p.stat().st_size, mtime=p.stat().st_mtime)
    cat.close()
    return cat_path


def test_geolocate_runs_with_mock(tmp_path: Path):
    photo_root = _make_photo_root(tmp_path)
    _seed_catalog(photo_root)

    result = runner.invoke(app, ["geolocate", str(photo_root), "--geolocator", "mock", "--top-k", "3"])
    assert result.exit_code == 0, result.output
    assert "processed=2" in result.output

    cat = Catalog(photo_root / ".photoindex" / "catalog.db")
    cat.init_schema()
    cur = cat._conn.execute("SELECT COUNT(*) FROM geo_predictions")  # noqa: SLF001
    assert cur.fetchone()[0] == 6  # 2 photos × top_k=3
    cat.close()


def test_geolocate_skips_already_predicted(tmp_path: Path):
    photo_root = _make_photo_root(tmp_path)
    _seed_catalog(photo_root)

    runner.invoke(app, ["geolocate", str(photo_root), "--geolocator", "mock", "--top-k", "3"])
    result = runner.invoke(app, ["geolocate", str(photo_root), "--geolocator", "mock", "--top-k", "3"])
    assert result.exit_code == 0
    assert "skipped=2" in result.output
    assert "processed=0" in result.output


def test_geolocate_force_repredicts(tmp_path: Path):
    photo_root = _make_photo_root(tmp_path)
    _seed_catalog(photo_root)

    runner.invoke(app, ["geolocate", str(photo_root), "--geolocator", "mock", "--top-k", "3"])
    result = runner.invoke(
        app, ["geolocate", str(photo_root), "--geolocator", "mock", "--top-k", "3", "--force"]
    )
    assert result.exit_code == 0
    assert "processed=2" in result.output


def test_geolocate_help_lists_choices(tmp_path: Path):
    result = runner.invoke(app, ["geolocate", "--help"])
    assert result.exit_code == 0
    assert "--geolocator" in result.output
    assert "geoclip" in result.output
    assert "mock" in result.output


def test_geolocate_errors_without_catalog(tmp_path: Path):
    photo_root = _make_photo_root(tmp_path)
    result = runner.invoke(app, ["geolocate", str(photo_root), "--geolocator", "mock"])
    assert result.exit_code != 0
    assert "no catalog at" in result.output


def test_export_zips_photoindex(tmp_path: Path):
    photo_root = _make_photo_root(tmp_path)
    _seed_catalog(photo_root)
    out_zip = tmp_path / "exported.zip"

    result = runner.invoke(app, ["export", str(photo_root), "--out", str(out_zip)])
    assert result.exit_code == 0, result.output
    assert out_zip.exists()

    with zipfile.ZipFile(out_zip) as zf:
        names = zf.namelist()
    assert any(n.endswith("catalog.db") for n in names)


def test_export_excludes_thumbs_by_default(tmp_path: Path):
    photo_root = _make_photo_root(tmp_path)
    _seed_catalog(photo_root)

    thumbs = photo_root / ".photoindex" / "thumbs"
    thumbs.mkdir()
    (thumbs / "thumb.jpg").write_bytes(b"thumb-data")

    out_zip = tmp_path / "exported.zip"
    runner.invoke(app, ["export", str(photo_root), "--out", str(out_zip)])

    with zipfile.ZipFile(out_zip) as zf:
        names = zf.namelist()
    assert not any("thumbs/" in n or "thumbs\\" in n for n in names)


def test_export_includes_thumbs_with_flag(tmp_path: Path):
    photo_root = _make_photo_root(tmp_path)
    _seed_catalog(photo_root)

    thumbs = photo_root / ".photoindex" / "thumbs"
    thumbs.mkdir()
    (thumbs / "thumb.jpg").write_bytes(b"thumb-data")

    out_zip = tmp_path / "exported.zip"
    runner.invoke(app, ["export", str(photo_root), "--out", str(out_zip), "--include-thumbs"])

    with zipfile.ZipFile(out_zip) as zf:
        names = zf.namelist()
    assert any("thumb.jpg" in n for n in names)
