from __future__ import annotations

from pathlib import Path

from PIL import Image

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

LONG_EDGE_TARGET = 1024

NON_RAW_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".tif", ".tiff", ".heic", ".heif", ".png",
})

RAW_EXTENSIONS: frozenset[str] = frozenset({
    ".arw", ".cr2", ".cr3", ".nef", ".raf", ".orf", ".rw2", ".dng",
})


def load_image(path: Path) -> Image.Image:
    ext = path.suffix.lower()
    if ext in NON_RAW_EXTENSIONS:
        img = Image.open(path)
    elif ext in RAW_EXTENSIONS:
        img = _load_raw(path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")
    img = img.convert("RGB")
    return _resize_long_edge(img, LONG_EDGE_TARGET)


def _load_raw(path: Path) -> Image.Image:
    import rawpy
    with rawpy.imread(str(path)) as raw:
        try:
            thumb = raw.extract_thumb()
        except rawpy.LibRawNoThumbnailError:
            # fall back: develop the raw (slow, but ensures we can always load something)
            rgb = raw.postprocess(no_auto_bright=True, output_bps=8)
            return Image.fromarray(rgb, mode="RGB")
    if thumb.format == rawpy.ThumbFormat.JPEG:
        from io import BytesIO
        return Image.open(BytesIO(thumb.data))
    # rawpy.ThumbFormat.BITMAP
    return Image.fromarray(thumb.data, mode="RGB")


def _resize_long_edge(img: Image.Image, target: int) -> Image.Image:
    w, h = img.size
    long_edge = max(w, h)
    if long_edge <= target:
        return img
    scale = target / long_edge
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)
