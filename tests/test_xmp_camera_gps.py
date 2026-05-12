from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from pixsage.xmp import CameraGps, read_camera_gps

EXIFTOOL = shutil.which("exiftool")
needs_exiftool = pytest.mark.skipif(EXIFTOOL is None, reason="exiftool not on PATH")


def _inject_exif_gps(
    path: Path,
    lat: float,
    lon: float,
    alt: float | None = None,
) -> None:
    """Write EXIF (not XMP) GPS tags into an existing image via exiftool."""
    lat_ref = "N" if lat >= 0 else "S"
    lon_ref = "E" if lon >= 0 else "W"
    args = [
        EXIFTOOL,
        "-overwrite_original",
        f"-EXIF:GPSLatitude={abs(lat)}",
        f"-EXIF:GPSLatitudeRef={lat_ref}",
        f"-EXIF:GPSLongitude={abs(lon)}",
        f"-EXIF:GPSLongitudeRef={lon_ref}",
    ]
    if alt is not None:
        args.extend([f"-EXIF:GPSAltitude={alt}", "-EXIF:GPSAltitudeRef=0"])
    args.append(str(path))
    subprocess.run(args, check=True, capture_output=True)


@needs_exiftool
def test_read_camera_gps_jpeg_with_full_gps(tmp_path: Path):
    p = tmp_path / "a.jpg"
    Image.new("RGB", (64, 64), color="red").save(p)
    _inject_exif_gps(p, lat=-52.4873, lon=-60.4447, alt=30.5)

    got = read_camera_gps(p)
    assert got is not None
    assert isinstance(got, CameraGps)
    assert abs(got.latitude - -52.4873) < 1e-3
    assert abs(got.longitude - -60.4447) < 1e-3
    assert got.altitude is not None
    assert abs(got.altitude - 30.5) < 1e-1


@needs_exiftool
def test_read_camera_gps_returns_none_when_absent(tmp_path: Path):
    p = tmp_path / "a.jpg"
    Image.new("RGB", (64, 64), color="red").save(p)
    assert read_camera_gps(p) is None


@needs_exiftool
def test_read_camera_gps_filters_zero_zero_sentinel(tmp_path: Path):
    """(0,0) is the 'no fix' bug from some old devices — return None."""
    p = tmp_path / "a.jpg"
    Image.new("RGB", (64, 64), color="red").save(p)
    _inject_exif_gps(p, lat=0.0, lon=0.0, alt=None)
    assert read_camera_gps(p) is None


@needs_exiftool
def test_read_camera_gps_without_altitude(tmp_path: Path):
    p = tmp_path / "a.jpg"
    Image.new("RGB", (64, 64), color="red").save(p)
    _inject_exif_gps(p, lat=51.5, lon=-0.1, alt=None)

    got = read_camera_gps(p)
    assert got is not None
    assert abs(got.latitude - 51.5) < 1e-3
    assert abs(got.longitude - -0.1) < 1e-3
    assert got.altitude is None


@needs_exiftool
def test_read_camera_gps_southern_hemisphere(tmp_path: Path):
    """Confirm S/W refs round-trip into signed decimal."""
    p = tmp_path / "a.jpg"
    Image.new("RGB", (64, 64), color="red").save(p)
    _inject_exif_gps(p, lat=-64.28, lon=-56.74, alt=None)

    got = read_camera_gps(p)
    assert got is not None
    assert got.latitude < 0
    assert got.longitude < 0
