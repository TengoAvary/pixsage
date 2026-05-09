from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from pixsage.web.thumbs import ThumbnailCache, ThumbSize


@pytest.fixture
def cache(tmp_path: Path) -> ThumbnailCache:
    return ThumbnailCache(root=tmp_path / "thumbs")


def test_get_or_create_returns_path_and_writes_file(cache: ThumbnailCache, tmp_path: Path):
    src = tmp_path / "a.jpg"
    Image.new("RGB", (1024, 768), color="red").save(src, "JPEG")

    path = cache.get_or_create("sha-a", src, ThumbSize.MEDIUM)
    assert path.exists()
    img = Image.open(path)
    assert max(img.size) == 720


def test_second_call_uses_cache(cache: ThumbnailCache, tmp_path: Path):
    src = tmp_path / "a.jpg"
    Image.new("RGB", (1024, 768), color="red").save(src, "JPEG")

    p1 = cache.get_or_create("sha-a", src, ThumbSize.SMALL)
    mtime1 = p1.stat().st_mtime
    p2 = cache.get_or_create("sha-a", src, ThumbSize.SMALL)
    assert p1 == p2
    assert p2.stat().st_mtime == mtime1


def test_different_sizes_create_different_files(cache: ThumbnailCache, tmp_path: Path):
    src = tmp_path / "a.jpg"
    Image.new("RGB", (1024, 768), color="red").save(src, "JPEG")

    s = cache.get_or_create("sha-a", src, ThumbSize.SMALL)
    m = cache.get_or_create("sha-a", src, ThumbSize.MEDIUM)
    l = cache.get_or_create("sha-a", src, ThumbSize.LARGE)

    assert s != m != l
    assert max(Image.open(s).size) == 256
    assert max(Image.open(m).size) == 720
    assert max(Image.open(l).size) == 1440


def test_path_uses_sha_prefix_for_dir_sharding(cache: ThumbnailCache, tmp_path: Path):
    src = tmp_path / "a.jpg"
    Image.new("RGB", (256, 256), color="red").save(src, "JPEG")
    sha = "abcd" + "0" * 60
    path = cache.get_or_create(sha, src, ThumbSize.SMALL)
    # Path shape: <root>/<size>/<sha[:2]>/<sha>.jpg
    assert path.parent.name == "ab"
    assert path.parent.parent.name == "small"
