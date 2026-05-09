from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from pixsage.images import LONG_EDGE_TARGET, load_image


def test_load_jpeg_returns_rgb(make_jpeg):
    p = make_jpeg("a.jpg", size=(2000, 1500), color="green")
    img = load_image(p)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_load_image_resizes_long_edge(make_jpeg):
    p = make_jpeg("big.jpg", size=(4000, 1000))
    img = load_image(p)
    assert max(img.size) == LONG_EDGE_TARGET


def test_load_image_preserves_aspect_ratio(make_jpeg):
    p = make_jpeg("wide.jpg", size=(2000, 500))
    img = load_image(p)
    w, h = img.size
    assert abs(w / h - 2000 / 500) < 0.02


def test_load_image_does_not_upscale(make_jpeg):
    p = make_jpeg("small.jpg", size=(400, 300))
    img = load_image(p)
    assert img.size == (400, 300)


def test_load_image_unknown_extension_raises(tmp_path: Path):
    p = tmp_path / "x.xyz"
    p.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        load_image(p)
