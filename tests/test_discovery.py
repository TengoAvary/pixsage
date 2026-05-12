from __future__ import annotations
from pathlib import Path

import pytest

from pixsage.discovery import walk_for_photoindex


def _make_catalog_dir(p: Path) -> None:
    """Make `p/.photoindex/` look like a real catalog dir."""
    (p / ".photoindex").mkdir(parents=True, exist_ok=True)
    (p / ".photoindex" / "catalog.db").write_bytes(b"")  # presence only


def test_walk_finds_top_level_photoindex(tmp_path: Path) -> None:
    _make_catalog_dir(tmp_path / "Sony")
    found = walk_for_photoindex([tmp_path], max_depth=6, time_budget_s=5)
    assert len(found) == 1
    assert found[0] == (tmp_path / "Sony" / ".photoindex").resolve()


def test_walk_finds_multiple_photoindex(tmp_path: Path) -> None:
    _make_catalog_dir(tmp_path / "Sony")
    _make_catalog_dir(tmp_path / "iPhone")
    found = walk_for_photoindex([tmp_path], max_depth=6, time_budget_s=5)
    assert len(found) == 2
    paths = sorted(str(p) for p in found)
    assert any("Sony" in p for p in paths)
    assert any("iPhone" in p for p in paths)


def test_walk_stops_descending_into_indexed_dirs(tmp_path: Path) -> None:
    """Once we find a .photoindex/, we don't keep looking inside that subtree."""
    _make_catalog_dir(tmp_path / "Sony")
    # A bogus nested .photoindex that should NOT be returned.
    nested = tmp_path / "Sony" / "Subfolder"
    _make_catalog_dir(nested)
    found = walk_for_photoindex([tmp_path], max_depth=6, time_budget_s=5)
    assert len(found) == 1
    assert "Subfolder" not in str(found[0])


def test_walk_respects_max_depth(tmp_path: Path) -> None:
    """A catalog 4 levels deep is found with depth=4, missed with depth=3."""
    deep = tmp_path / "a" / "b" / "c" / "d"
    _make_catalog_dir(deep)
    # depth=4 walks tmp_path -> a -> b -> c -> d (d gets the find)
    assert len(walk_for_photoindex([tmp_path], max_depth=4, time_budget_s=5)) == 1
    assert len(walk_for_photoindex([tmp_path], max_depth=3, time_budget_s=5)) == 0


def test_walk_skips_hidden_directories(tmp_path: Path) -> None:
    """Don't descend into .git, node_modules, etc."""
    _make_catalog_dir(tmp_path / ".git")  # hidden — should be skipped
    _make_catalog_dir(tmp_path / "Sony")
    found = walk_for_photoindex([tmp_path], max_depth=6, time_budget_s=5)
    assert len(found) == 1
    assert "Sony" in str(found[0])


def test_walk_handles_missing_root(tmp_path: Path) -> None:
    """A root that doesn't exist is silently skipped, not raised."""
    found = walk_for_photoindex([tmp_path / "doesnotexist"], max_depth=6, time_budget_s=5)
    assert found == []


def test_walk_handles_permission_error(tmp_path: Path, monkeypatch) -> None:
    """A directory we can't read is logged and skipped, not raised."""
    _make_catalog_dir(tmp_path / "Sony")
    real_iterdir = Path.iterdir

    def fake_iterdir(self):
        if self.name == "denied":
            raise PermissionError("nope")
        return real_iterdir(self)

    (tmp_path / "denied").mkdir()
    monkeypatch.setattr(Path, "iterdir", fake_iterdir)

    found = walk_for_photoindex([tmp_path], max_depth=6, time_budget_s=5)
    # Sony still found despite denied dir
    assert len(found) == 1
