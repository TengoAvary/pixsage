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
