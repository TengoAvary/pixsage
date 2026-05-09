from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def make_jpeg(tmp_path: Path):
    """Factory: write a synthetic JPEG to tmp_path/<name>.jpg, return the path."""
    def _make(name: str = "img.jpg", size: tuple[int, int] = (800, 600), color: str = "red") -> Path:
        path = tmp_path / name
        img = Image.new("RGB", size, color=color)
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
