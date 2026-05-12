from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from pixsage.xmp import CameraGps, XmpFields, read_camera_gps, read_metadata, write_xmp

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


@needs_exiftool
def test_read_metadata_jpeg_returns_both_xmp_and_gps(tmp_path: Path):
    """For an embedded-XMP file, one subprocess fetches both XMP and EXIF GPS."""
    p = tmp_path / "a.jpg"
    Image.new("RGB", (64, 64), color="red").save(p)
    write_xmp(p, XmpFields(subject=["foo"], hierarchical_subject=[], description="hello"), is_raw=False)
    _inject_exif_gps(p, lat=10.0, lon=20.0, alt=100.0)

    xmp_fields, gps = read_metadata(p, is_raw=False)
    assert xmp_fields.subject == ["foo"]
    assert xmp_fields.description == "hello"
    assert gps is not None
    assert abs(gps.latitude - 10.0) < 1e-3
    assert abs(gps.longitude - 20.0) < 1e-3


@needs_exiftool
def test_read_metadata_jpeg_without_gps(tmp_path: Path):
    p = tmp_path / "a.jpg"
    Image.new("RGB", (64, 64), color="red").save(p)
    write_xmp(p, XmpFields(subject=["foo"], hierarchical_subject=[], description=None), is_raw=False)

    xmp_fields, gps = read_metadata(p, is_raw=False)
    assert xmp_fields.subject == ["foo"]
    assert gps is None


@needs_exiftool
def test_read_metadata_raw_sidecar(tmp_path: Path):
    """For sidecar raws, XMP comes from the sidecar and GPS from the raw file.

    Uses a placeholder raw + empty sidecar (the same pattern as
    test_xmp_gps.test_write_gps_updates_existing_sidecar_for_raw_path).
    """
    raw = tmp_path / "DSC0001.arw"
    # Write a valid JPEG masquerading as .arw so exiftool can read EXIF GPS off it.
    # exiftool rejects writing EXIF to a mismatched extension, so inject GPS
    # into a .jpg first, then rename to .arw — exiftool reads it fine either way.
    jpg = tmp_path / "DSC0001.jpg"
    Image.new("RGB", (64, 64), color="red").save(jpg, format="JPEG")
    _inject_exif_gps(jpg, lat=-1.0, lon=2.0, alt=None)
    jpg.rename(raw)

    sidecar = raw.with_suffix(".xmp")
    write_xmp(raw, XmpFields(subject=["bar"], hierarchical_subject=[], description="x"), is_raw=True)
    assert sidecar.exists()

    xmp_fields, gps = read_metadata(raw, is_raw=True)
    assert xmp_fields.subject == ["bar"]
    assert xmp_fields.description == "x"
    assert gps is not None
    assert abs(gps.latitude - -1.0) < 1e-3
    assert abs(gps.longitude - 2.0) < 1e-3
