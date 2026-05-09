from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def make_jpeg(tmp_path: Path):
    """Factory: write a synthetic JPEG to tmp_path/<name>.jpg, return the path.

    Each call produces unique bytes by drawing a tiny pattern derived from
    `name` so identical-looking JPEGs from sequential factory calls don't
    collapse to the same sha256.
    """
    def _make(name: str = "img.jpg", size: tuple[int, int] = (800, 600), color: str = "red") -> Path:
        path = tmp_path / name
        img = Image.new("RGB", size, color=color)
        # Stamp a single off-color pixel at a name-derived location so each
        # JPEG hashes differently even if size/color match.
        h = abs(hash(name))
        x = h % size[0]
        y = (h // size[0]) % size[1]
        img.putpixel((x, y), ((h >> 16) & 0xff, (h >> 8) & 0xff, h & 0xff))
        img.save(path, format="JPEG", quality=85)
        return path
    return _make


@pytest.fixture
def photo_root(tmp_path: Path) -> Path:
    """An empty photo root with a .photoindex/ subdirectory."""
    root = tmp_path / "photos"
    root.mkdir()
    (root / ".photoindex").mkdir()
    return root
