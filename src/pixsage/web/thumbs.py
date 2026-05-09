from __future__ import annotations

from enum import Enum
from pathlib import Path

from pixsage.images import load_image


class ThumbSize(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


_LONG_EDGE = {
    ThumbSize.SMALL: 256,
    ThumbSize.MEDIUM: 720,
    ThumbSize.LARGE: 1440,
}


class ThumbnailCache:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, sha256: str, size: ThumbSize) -> Path:
        return self.root / size.value / sha256[:2] / f"{sha256}.jpg"

    def get_or_create(self, sha256: str, source: Path, size: ThumbSize) -> Path:
        out = self.path_for(sha256, size)
        if out.exists():
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        img = load_image(source).convert("RGB")
        long_edge = _LONG_EDGE[size]
        if max(img.size) != long_edge:
            ratio = long_edge / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size)
        img.save(out, "JPEG", quality=85)
        return out
