"""User-scoped catalog registry persisted to <runtime>/catalogs.json.

Owned by the serve process. Tracks every catalog the app has ever seen
plus the user's enable/disable choice per catalog. Discovery (in
discovery.py) feeds new paths into the registry; the web UI mutates it
via the routes added in tests/test_web_catalogs.py.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator


REGISTRY_VERSION = 1


@dataclass
class CatalogEntry:
    id: str
    photoindex_path: str
    label: str
    enabled: bool
    first_seen: str
    last_seen: str
    image_embedder_signature: str | None
    caption_embedder_signature: str | None
    # Not persisted — derived at load time by Registry.refresh_availability().
    available: bool = field(default=False, compare=False)


class Registry:
    """JSON-backed catalog registry. Single-writer per process."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._entries: list[CatalogEntry] = []

    def load(self) -> None:
        """Read the registry file. Empty list if missing. Corrupt file is
        backed up to <path>.broken-<ts> and replaced with an empty registry."""
        if not self.path.exists():
            self._entries = []
            return
        raw = self.path.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            backup = self.path.with_name(f"{self.path.name}.broken-{int(time.time())}")
            shutil.move(str(self.path), str(backup))
            self._entries = []
            return
        version = data.get("version")
        if version != REGISTRY_VERSION:
            raise RuntimeError(
                f"unsupported registry version {version!r} at {self.path}; expected {REGISTRY_VERSION}"
            )
        self._entries = [CatalogEntry(**c) for c in data.get("catalogs", [])]

    def save(self) -> None:
        """Persist current entries. Strips the non-persisted `available` field."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": REGISTRY_VERSION,
            "catalogs": [
                {k: v for k, v in asdict(e).items() if k != "available"}
                for e in self._entries
            ],
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def entries(self) -> Iterator[CatalogEntry]:
        return iter(self._entries)

    def add(
        self,
        photoindex_path: str,
        label: str,
        image_embedder_signature: str | None,
        caption_embedder_signature: str | None,
        enabled: bool = True,
    ) -> CatalogEntry:
        """Add a new catalog. Generates an id. Toggled on by default."""
        import uuid
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        entry = CatalogEntry(
            id=uuid.uuid4().hex,
            photoindex_path=photoindex_path,
            label=label,
            enabled=enabled,
            first_seen=now,
            last_seen=now,
            image_embedder_signature=image_embedder_signature,
            caption_embedder_signature=caption_embedder_signature,
        )
        self._entries.append(entry)
        return entry

    def find_by_id(self, id: str) -> CatalogEntry | None:
        for e in self._entries:
            if e.id == id:
                return e
        return None

    def find_by_photoindex_path(self, path: str) -> CatalogEntry | None:
        # Compare resolved + normalised paths so /a/./b matches /a/b
        target = str(Path(path).resolve())
        for e in self._entries:
            if str(Path(e.photoindex_path).resolve()) == target:
                return e
        return None

    def toggle(self, id: str) -> None:
        e = self.find_by_id(id)
        if e is None:
            raise KeyError(f"no catalog with id {id!r}")
        e.enabled = not e.enabled

    def rename(self, id: str, label: str) -> None:
        e = self.find_by_id(id)
        if e is None:
            raise KeyError(f"no catalog with id {id!r}")
        e.label = label

    def remove(self, id: str) -> None:
        for i, e in enumerate(self._entries):
            if e.id == id:
                del self._entries[i]
                return
        raise KeyError(f"no catalog with id {id!r}")

    def mark_available(self, id: str, available: bool) -> None:
        e = self.find_by_id(id)
        if e is None:
            raise KeyError(f"no catalog with id {id!r}")
        e.available = available

    def refresh_from_discovery(self, discovered_paths: list[Path]) -> None:
        """Reconcile the registry against the filesystem.

        For each existing entry: set `available` based on whether its
        photoindex_path exists.

        For each discovered path not yet in the registry: add it (toggled on)
        and mark it available. The label defaults to the parent directory's
        name (e.g. /Volumes/Sony/.photoindex -> "Sony").
        """
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

        # Step 1: refresh availability for known entries.
        for e in self._entries:
            e.available = Path(e.photoindex_path).exists()
            if e.available:
                e.last_seen = now

        # Step 2: auto-add new discoveries.
        for p in discovered_paths:
            p = Path(p).resolve()
            if self.find_by_photoindex_path(str(p)) is not None:
                continue
            label = p.parent.name  # /Volumes/Sony/.photoindex -> "Sony"
            img_sig, cap_sig = derive_signatures(p)
            entry = self.add(
                photoindex_path=str(p),
                label=label,
                image_embedder_signature=img_sig,
                caption_embedder_signature=cap_sig,
                enabled=True,
            )
            entry.available = True


# Default signatures used when a catalog's meta doesn't record them.
# Matches what pixsage currently embeds with: SigLIP2-so400m + MiniLM-L6-v2.
DEFAULT_IMAGE_SIGNATURE = "siglip2-so400m-patch14-384@v1"
DEFAULT_CAPTION_SIGNATURE = "minilm-L6-v2@v2"


def derive_signatures(photoindex_path: Path | str) -> tuple[str, str]:
    """Read (image_signature, caption_signature) from a catalog.

    `photoindex_path` is the `.photoindex/` directory; this function looks
    inside it for `catalog.db` and reads the `image_embedder_signature` /
    `caption_embedder_signature` meta keys. Falls back to DEFAULT_* if either
    the file is missing or the meta key is unset.

    Order:
    1. Catalog meta keys `image_embedder_signature` / `caption_embedder_signature`
       (written by `pixsage embed` for new catalogs).
    2. DEFAULT_* constants (for catalogs embedded before this feature shipped).
    """
    from pixsage.catalog import Catalog
    catalog_path = Path(photoindex_path) / "catalog.db"
    if not catalog_path.exists():
        return DEFAULT_IMAGE_SIGNATURE, DEFAULT_CAPTION_SIGNATURE
    cat = Catalog(catalog_path)
    img = cat.get_meta("image_embedder_signature") or DEFAULT_IMAGE_SIGNATURE
    cap = cat.get_meta("caption_embedder_signature") or DEFAULT_CAPTION_SIGNATURE
    return img, cap
