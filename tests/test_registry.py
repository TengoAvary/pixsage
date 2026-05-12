from __future__ import annotations
import json
from pathlib import Path

import pytest

from pixsage.registry import CatalogEntry, Registry, REGISTRY_VERSION


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "catalogs.json")
    reg.load()
    assert list(reg.entries()) == []


def test_save_then_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "catalogs.json"
    reg = Registry(path)
    reg.load()
    entry = CatalogEntry(
        id="abc123",
        photoindex_path="/Volumes/Sony/.photoindex",
        label="Sony",
        enabled=True,
        first_seen="2026-05-12T14:00:00Z",
        last_seen="2026-05-12T14:00:00Z",
        image_embedder_signature="siglip2@v1",
        caption_embedder_signature="minilm@v2",
    )
    reg._entries.append(entry)
    reg.save()

    reg2 = Registry(path)
    reg2.load()
    loaded = list(reg2.entries())
    assert len(loaded) == 1
    assert loaded[0].id == "abc123"
    assert loaded[0].photoindex_path == "/Volumes/Sony/.photoindex"
    assert loaded[0].enabled is True


def test_load_corrupt_json_backs_up_and_starts_fresh(tmp_path: Path) -> None:
    path = tmp_path / "catalogs.json"
    path.write_text("this is not json", encoding="utf-8")
    reg = Registry(path)
    reg.load()
    assert list(reg.entries()) == []
    backups = list(tmp_path.glob("catalogs.json.broken-*"))
    assert len(backups) == 1
    assert backups[0].read_text() == "this is not json"


def test_load_unknown_version_raises(tmp_path: Path) -> None:
    path = tmp_path / "catalogs.json"
    path.write_text(json.dumps({"version": 999, "catalogs": []}), encoding="utf-8")
    reg = Registry(path)
    with pytest.raises(RuntimeError, match="unsupported registry version"):
        reg.load()


def test_save_writes_version_field(tmp_path: Path) -> None:
    path = tmp_path / "catalogs.json"
    reg = Registry(path)
    reg.load()
    reg.save()
    data = json.loads(path.read_text())
    assert data["version"] == REGISTRY_VERSION
    assert data["catalogs"] == []
