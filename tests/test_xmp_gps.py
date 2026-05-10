from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PIL import Image

from pixsage.xmp import read_gps, write_gps


EXIFTOOL = shutil.which("exiftool")
needs_exiftool = pytest.mark.skipif(EXIFTOOL is None, reason="exiftool not on PATH")


@needs_exiftool
def test_write_and_read_gps_jpeg(tmp_path: Path):
    p = tmp_path / "a.jpg"
    Image.new("RGB", (64, 64), color="red").save(p)
    write_gps(p, -64.2799, -56.7449, "Seymour Island", is_raw=False)
    got = read_gps(p, is_raw=False)
    assert got is not None
    assert abs(got["latitude"] - -64.2799) < 1e-4
    assert abs(got["longitude"] - -56.7449) < 1e-4
    assert got["place_name"] == "Seymour Island"


@needs_exiftool
def test_write_gps_positive_hemispheres(tmp_path: Path):
    p = tmp_path / "a.jpg"
    Image.new("RGB", (64, 64), color="red").save(p)
    write_gps(p, 51.5074, -0.1278, "London", is_raw=False)
    got = read_gps(p, is_raw=False)
    assert abs(got["latitude"] - 51.5074) < 1e-4
    assert abs(got["longitude"] - -0.1278) < 1e-4


@needs_exiftool
def test_write_gps_no_place_name(tmp_path: Path):
    p = tmp_path / "a.jpg"
    Image.new("RGB", (64, 64), color="red").save(p)
    write_gps(p, 0.0, 0.0, None, is_raw=False)
    got = read_gps(p, is_raw=False)
    assert got is not None
    assert got["place_name"] is None


@needs_exiftool
def test_read_gps_returns_none_when_absent(tmp_path: Path):
    p = tmp_path / "a.jpg"
    Image.new("RGB", (64, 64), color="red").save(p)
    assert read_gps(p, is_raw=False) is None


@needs_exiftool
def test_write_gps_updates_existing_sidecar_for_raw_path(tmp_path: Path):
    """For raw paths, GPS should write into the .xmp sidecar."""
    raw = tmp_path / "DSC0001.arw"
    raw.write_bytes(b"\x00")  # placeholder; we use an existing sidecar path
    sidecar = raw.with_suffix(".xmp")
    # Seed an empty XMP sidecar so write_gps takes the "update existing" branch
    # (avoids exiftool needing to parse the placeholder bytes as a real raw).
    sidecar.write_text(
        '<?xpacket begin=""?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description/></rdf:RDF></x:xmpmeta><?xpacket end=""?>',
        encoding="utf-8",
    )
    write_gps(raw, -64.0, -57.0, "Test", is_raw=True)
    got = read_gps(raw, is_raw=True)
    assert got is not None
    assert abs(got["latitude"] - -64.0) < 1e-4
    assert abs(got["longitude"] - -57.0) < 1e-4
    assert got["place_name"] == "Test"
