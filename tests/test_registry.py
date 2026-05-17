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


def test_add_assigns_id_and_returns_entry(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "catalogs.json")
    reg.load()
    entry = reg.add(
        photoindex_path="/Volumes/Sony/.photoindex",
        label="Sony",
        image_embedder_signature="siglip2@v1",
        caption_embedder_signature="minilm@v2",
    )
    assert entry.id  # non-empty
    assert entry.enabled is True
    assert entry.label == "Sony"
    assert reg.find_by_id(entry.id) is entry


def test_find_by_photoindex_path(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "catalogs.json")
    reg.load()
    e1 = reg.add(photoindex_path="/a/.photoindex", label="A",
                 image_embedder_signature="x", caption_embedder_signature="y")
    e2 = reg.add(photoindex_path="/b/.photoindex", label="B",
                 image_embedder_signature="x", caption_embedder_signature="y")
    assert reg.find_by_photoindex_path("/a/.photoindex") is e1
    assert reg.find_by_photoindex_path("/c/.photoindex") is None


def test_toggle_flips_enabled(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "catalogs.json")
    reg.load()
    e = reg.add(photoindex_path="/a/.photoindex", label="A",
                image_embedder_signature="x", caption_embedder_signature="y")
    assert e.enabled is True
    reg.toggle(e.id)
    assert reg.find_by_id(e.id).enabled is False
    reg.toggle(e.id)
    assert reg.find_by_id(e.id).enabled is True


def test_rename(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "catalogs.json")
    reg.load()
    e = reg.add(photoindex_path="/a/.photoindex", label="Old",
                image_embedder_signature="x", caption_embedder_signature="y")
    reg.rename(e.id, "New")
    assert reg.find_by_id(e.id).label == "New"


def test_remove(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "catalogs.json")
    reg.load()
    e1 = reg.add(photoindex_path="/a/.photoindex", label="A",
                 image_embedder_signature="x", caption_embedder_signature="y")
    e2 = reg.add(photoindex_path="/b/.photoindex", label="B",
                 image_embedder_signature="x", caption_embedder_signature="y")
    reg.remove(e1.id)
    assert reg.find_by_id(e1.id) is None
    assert reg.find_by_id(e2.id) is e2


def test_mark_available_updates_runtime_flag_only(tmp_path: Path) -> None:
    """available is runtime-only; mark_available must not persist."""
    path = tmp_path / "catalogs.json"
    reg = Registry(path)
    reg.load()
    e = reg.add(photoindex_path="/a/.photoindex", label="A",
                image_embedder_signature="x", caption_embedder_signature="y")
    reg.mark_available(e.id, True)
    assert reg.find_by_id(e.id).available is True
    reg.save()
    data = json.loads(path.read_text())
    assert "available" not in data["catalogs"][0]


def test_find_by_id_missing_returns_none(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "catalogs.json")
    reg.load()
    assert reg.find_by_id("nonexistent") is None


def test_remove_missing_id_raises(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "catalogs.json")
    reg.load()
    with pytest.raises(KeyError):
        reg.remove("nonexistent")


def test_derive_signatures_reads_meta(tmp_path: Path) -> None:
    """If catalog.meta has the signature keys, derive_signatures returns them."""
    from pixsage.catalog import Catalog
    from pixsage.registry import derive_signatures

    photoindex = tmp_path / ".photoindex"
    photoindex.mkdir()
    cat = Catalog(photoindex / "catalog.db")
    cat.init_schema()
    cat.set_meta("image_embedder_signature", "siglip2-so400m@v1")
    cat.set_meta("caption_embedder_signature", "minilm-L6-v2@v2")

    img, cap = derive_signatures(photoindex)
    assert img == "siglip2-so400m@v1"
    assert cap == "minilm-L6-v2@v2"


def test_derive_signatures_falls_back_to_defaults(tmp_path: Path) -> None:
    """Old catalogs with no signature meta get the codebase's default signatures."""
    from pixsage.catalog import Catalog
    from pixsage.registry import (
        DEFAULT_IMAGE_SIGNATURE,
        DEFAULT_CAPTION_SIGNATURE,
        derive_signatures,
    )

    photoindex = tmp_path / ".photoindex"
    photoindex.mkdir()
    cat = Catalog(photoindex / "catalog.db")
    cat.init_schema()
    # No meta keys set.

    img, cap = derive_signatures(photoindex)
    assert img == DEFAULT_IMAGE_SIGNATURE
    assert cap == DEFAULT_CAPTION_SIGNATURE


def test_refresh_marks_existing_available(tmp_path: Path) -> None:
    """A registered entry whose photoindex_path exists is marked available."""
    photoindex = tmp_path / "Sony" / ".photoindex"
    photoindex.mkdir(parents=True)
    (photoindex / "catalog.db").write_bytes(b"")

    reg = Registry(tmp_path / "catalogs.json")
    reg.load()
    e = reg.add(
        photoindex_path=str(photoindex.resolve()),
        label="Sony",
        image_embedder_signature="x",
        caption_embedder_signature="y",
    )
    reg.refresh_from_discovery(discovered_paths=[])
    assert reg.find_by_id(e.id).available is True


def test_refresh_marks_missing_offline(tmp_path: Path) -> None:
    """A registered entry whose path doesn't exist is marked offline."""
    reg = Registry(tmp_path / "catalogs.json")
    reg.load()
    e = reg.add(
        photoindex_path="/Volumes/NotMounted/.photoindex",
        label="Offline",
        image_embedder_signature="x",
        caption_embedder_signature="y",
    )
    reg.refresh_from_discovery(discovered_paths=[])
    assert reg.find_by_id(e.id).available is False


def test_refresh_auto_adds_new_discoveries(tmp_path: Path, monkeypatch) -> None:
    """A discovered path that's not in the registry gets added, toggled on."""
    photoindex = tmp_path / "iPhone" / ".photoindex"
    photoindex.mkdir(parents=True)
    (photoindex / "catalog.db").write_bytes(b"")

    # Stub derive_signatures so we don't need a real catalog schema
    from pixsage import registry as registry_mod
    monkeypatch.setattr(
        registry_mod, "derive_signatures",
        lambda p: ("siglip2@v1", "minilm@v2"),
    )

    reg = Registry(tmp_path / "catalogs.json")
    reg.load()
    reg.refresh_from_discovery(discovered_paths=[photoindex.resolve()])

    entries = list(reg.entries())
    assert len(entries) == 1
    assert entries[0].enabled is True
    assert entries[0].available is True
    assert entries[0].label == "iPhone"  # derived from parent dir name


def test_refresh_does_not_duplicate_known_path(tmp_path: Path, monkeypatch) -> None:
    """Discovering an already-registered path is a no-op (no duplicate)."""
    photoindex = tmp_path / "Sony" / ".photoindex"
    photoindex.mkdir(parents=True)
    (photoindex / "catalog.db").write_bytes(b"")

    from pixsage import registry as registry_mod
    monkeypatch.setattr(
        registry_mod, "derive_signatures",
        lambda p: ("siglip2@v1", "minilm@v2"),
    )

    reg = Registry(tmp_path / "catalogs.json")
    reg.load()
    reg.add(
        photoindex_path=str(photoindex.resolve()),
        label="Sony",
        image_embedder_signature="x",
        caption_embedder_signature="y",
    )
    reg.refresh_from_discovery(discovered_paths=[photoindex.resolve()])
    assert len(list(reg.entries())) == 1


def test_refresh_availability_marks_existing(tmp_path: Path) -> None:
    pi = tmp_path / "Sony" / ".photoindex"
    pi.mkdir(parents=True)
    reg = Registry(tmp_path / "catalogs.json")
    reg.load()
    reg.add(photoindex_path=str(pi.resolve()), label="Sony",
            image_embedder_signature="i", caption_embedder_signature="c")
    before = list(reg.entries())[0].last_seen
    reg.refresh_availability()
    e = list(reg.entries())[0]
    assert e.available is True
    assert e.last_seen >= before
    assert e.last_seen.endswith("Z")


def test_refresh_availability_marks_offline_and_adds_nothing(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "catalogs.json")
    reg.load()
    reg.add(photoindex_path="/Volumes/NotMounted/.photoindex", label="Gone",
            image_embedder_signature="i", caption_embedder_signature="c")
    reg.refresh_availability()
    entries = list(reg.entries())
    assert len(entries) == 1          # never adds
    assert entries[0].available is False
