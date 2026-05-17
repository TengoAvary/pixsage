from __future__ import annotations

from pathlib import Path

from pixsage.walker import IMAGE_EXTENSIONS, sample_paths, sha256_file, walk_photos


def test_sha256_file_known_input(tmp_path: Path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello world")
    # known sha256("hello world")
    assert sha256_file(p) == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_walk_photos_finds_images(tmp_path: Path, make_jpeg):
    make_jpeg("a.jpg")
    make_jpeg("b.JPG")
    (tmp_path / "notes.txt").write_text("hi")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.tif").write_bytes(b"\x00")  # not a real tif but extension matches
    found = sorted(p.name.lower() for p in walk_photos(tmp_path))
    assert found == ["a.jpg", "b.jpg", "c.tif"]


def test_walk_photos_skips_photoindex_dir(tmp_path: Path, make_jpeg):
    make_jpeg("ok.jpg")
    idx = tmp_path / ".photoindex"
    idx.mkdir()
    # Even if a stray jpeg ends up under .photoindex/, we ignore it.
    Path(idx / "ignored.jpg").write_bytes(b"\xff\xd8\xff")
    found = [p.name for p in walk_photos(tmp_path)]
    assert found == ["ok.jpg"]


def test_walk_photos_skips_appledouble_files(tmp_path: Path, make_jpeg):
    """macOS writes ._<name> resource-fork stubs alongside real files on
    exFAT/FAT drives. They carry an image extension but are not images;
    skip them so they never reach the decoders."""
    make_jpeg("DSC0001.jpg")
    # AppleDouble companion — same extension, not an image.
    Path(tmp_path / "._DSC0001.jpg").write_bytes(b"\x00\x05\x16\x07not-an-image")
    Path(tmp_path / "._DSC0002.arw").write_bytes(b"\x00\x05\x16\x07not-an-image")
    found = [p.name for p in walk_photos(tmp_path)]
    assert found == ["DSC0001.jpg"]


def test_image_extensions_includes_common_raws_and_jpegs():
    assert ".jpg" in IMAGE_EXTENSIONS
    assert ".heic" in IMAGE_EXTENSIONS
    assert ".arw" in IMAGE_EXTENSIONS
    assert ".cr3" in IMAGE_EXTENSIONS
    assert ".nef" in IMAGE_EXTENSIONS
    assert ".dng" in IMAGE_EXTENSIONS


def test_sample_paths_deterministic(tmp_path: Path):
    paths = [tmp_path / f"{i:03d}.jpg" for i in range(20)]
    hashes = {p: f"{i:064x}" for i, p in enumerate(paths)}
    sampled1 = sample_paths(paths, hashes, n=5)
    sampled2 = sample_paths(paths, hashes, n=5)
    assert sampled1 == sampled2
    assert len(sampled1) == 5


def test_sample_paths_caps_at_total(tmp_path: Path):
    paths = [tmp_path / f"{i}.jpg" for i in range(3)]
    hashes = {p: f"{i:064x}" for i, p in enumerate(paths)}
    assert len(sample_paths(paths, hashes, n=99)) == 3
