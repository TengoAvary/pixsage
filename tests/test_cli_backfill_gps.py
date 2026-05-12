from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from pixsage.catalog import Catalog
from pixsage.cli import app

EXIFTOOL = shutil.which("exiftool")
needs_exiftool = pytest.mark.skipif(EXIFTOOL is None, reason="exiftool not on PATH")


def _inject_exif_gps(path: Path, lat: float, lon: float) -> None:
    args = [
        EXIFTOOL,
        "-overwrite_original",
        f"-EXIF:GPSLatitude={abs(lat)}",
        f"-EXIF:GPSLatitudeRef={'N' if lat >= 0 else 'S'}",
        f"-EXIF:GPSLongitude={abs(lon)}",
        f"-EXIF:GPSLongitudeRef={'E' if lon >= 0 else 'W'}",
        str(path),
    ]
    subprocess.run(args, check=True, capture_output=True)


@needs_exiftool
def test_backfill_populates_camera_gps_for_existing_catalog(tmp_path: Path):
    photo_root = tmp_path / "corpus"
    photo_root.mkdir()
    photoindex = photo_root / ".photoindex"
    photoindex.mkdir()
    cat = Catalog(photoindex / "catalog.db")
    cat.init_schema()

    # Simulate a previous tag run: catalog has rows but no exif_latitude yet.
    img1 = photo_root / "a.jpg"
    img2 = photo_root / "b.jpg"
    Image.new("RGB", (64, 64), color="red").save(img1)
    Image.new("RGB", (64, 64), color="blue").save(img2)
    _inject_exif_gps(img1, lat=-52.5, lon=-60.4)
    # img2 deliberately has no GPS.

    cat.upsert_photo("sha-a", img1, filesize=img1.stat().st_size, mtime=img1.stat().st_mtime)
    cat.upsert_photo("sha-b", img2, filesize=img2.stat().st_size, mtime=img2.stat().st_mtime)
    cat.close()

    runner = CliRunner()
    result = runner.invoke(app, ["backfill-exif-gps", str(photo_root)])
    assert result.exit_code == 0, result.output
    assert "checked: 2" in result.output.lower()
    assert "with gps: 1" in result.output.lower()

    cat = Catalog(photoindex / "catalog.db")
    cat.init_schema()
    assert cat.get_camera_gps("sha-a") is not None
    assert cat.get_camera_gps("sha-b") is None
    cat.close()


@needs_exiftool
def test_backfill_skips_already_populated_rows(tmp_path: Path):
    photo_root = tmp_path / "corpus"
    photo_root.mkdir()
    photoindex = photo_root / ".photoindex"
    photoindex.mkdir()
    cat = Catalog(photoindex / "catalog.db")
    cat.init_schema()

    img = photo_root / "a.jpg"
    Image.new("RGB", (64, 64), color="red").save(img)
    _inject_exif_gps(img, lat=-52.5, lon=-60.4)
    cat.upsert_photo("sha-a", img, filesize=img.stat().st_size, mtime=img.stat().st_mtime)
    cat.set_camera_gps("sha-a", latitude=99.0, longitude=99.0, altitude=None)
    cat.close()

    runner = CliRunner()
    result = runner.invoke(app, ["backfill-exif-gps", str(photo_root)])
    assert result.exit_code == 0, result.output
    assert "skipped: 1" in result.output.lower()

    # The pre-existing (99, 99) value should be untouched.
    cat = Catalog(photoindex / "catalog.db")
    cat.init_schema()
    got = cat.get_camera_gps("sha-a")
    assert got is not None
    assert got["latitude"] == 99.0
    cat.close()


@needs_exiftool
def test_backfill_force_refreshes_populated_rows(tmp_path: Path):
    photo_root = tmp_path / "corpus"
    photo_root.mkdir()
    photoindex = photo_root / ".photoindex"
    photoindex.mkdir()
    cat = Catalog(photoindex / "catalog.db")
    cat.init_schema()

    img = photo_root / "a.jpg"
    Image.new("RGB", (64, 64), color="red").save(img)
    _inject_exif_gps(img, lat=-52.5, lon=-60.4)
    cat.upsert_photo("sha-a", img, filesize=img.stat().st_size, mtime=img.stat().st_mtime)
    cat.set_camera_gps("sha-a", latitude=99.0, longitude=99.0, altitude=None)
    cat.close()

    runner = CliRunner()
    result = runner.invoke(app, ["backfill-exif-gps", str(photo_root), "--force"])
    assert result.exit_code == 0, result.output

    cat = Catalog(photoindex / "catalog.db")
    cat.init_schema()
    got = cat.get_camera_gps("sha-a")
    assert got is not None
    assert abs(got["latitude"] - -52.5) < 1e-3
    cat.close()


@needs_exiftool
def test_backfill_handles_missing_files_gracefully(tmp_path: Path):
    photo_root = tmp_path / "corpus"
    photo_root.mkdir()
    photoindex = photo_root / ".photoindex"
    photoindex.mkdir()
    cat = Catalog(photoindex / "catalog.db")
    cat.init_schema()

    # Reference a path that doesn't exist (simulates portable catalog).
    cat.upsert_photo(
        "sha-missing",
        Path("/nonexistent/path.jpg"),
        filesize=0,
        mtime=0.0,
    )
    cat.close()

    runner = CliRunner()
    result = runner.invoke(app, ["backfill-exif-gps", str(photo_root)])
    assert result.exit_code == 0, result.output
    assert "missing: 1" in result.output.lower()
