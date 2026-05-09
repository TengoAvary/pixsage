from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg", ".jpeg",
    ".tif", ".tiff",
    ".heic", ".heif",
    ".png",
    # raws
    ".arw", ".cr2", ".cr3", ".nef", ".raf", ".orf", ".rw2", ".dng",
})

PHOTOINDEX_DIR = ".photoindex"

CHUNK_SIZE = 1 << 20  # 1 MiB


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def walk_photos(root: Path) -> Iterator[Path]:
    """Yield every image file under root, skipping .photoindex/."""
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if PHOTOINDEX_DIR in p.parts:
            continue
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p


def sample_paths(paths: list[Path], hashes: dict[Path, str], n: int) -> list[Path]:
    """Deterministic sample: sort by sha256, take first n."""
    return sorted(paths, key=lambda p: hashes[p])[:n]
