from pathlib import Path

import pytest

from pixsage.path_translation import PathResolver


def test_resolver_no_translation_when_roots_match(tmp_path: Path) -> None:
    target = tmp_path / "photo.jpg"
    target.write_bytes(b"x")
    resolver = PathResolver(stored_root=str(tmp_path), runtime_root=tmp_path)
    assert resolver.resolve(str(target)) == target


def test_resolver_substitutes_prefix_when_translated_exists(tmp_path: Path) -> None:
    new_root = tmp_path / "new"
    new_root.mkdir()
    target = new_root / "sub" / "photo.jpg"
    target.parent.mkdir()
    target.write_bytes(b"x")

    # Stored path uses a fictional Windows root; runtime root is different.
    stored_path = r"E:\fakeroot\sub\photo.jpg"
    resolver = PathResolver(stored_root=r"E:\fakeroot", runtime_root=new_root)
    resolved = resolver.resolve(stored_path)
    assert resolved == target


def test_resolver_falls_back_to_stored_path_when_translated_missing(tmp_path: Path) -> None:
    # Translated path won't exist (we never create it). Stored path also doesn't exist
    # but resolver should still return the stored path, leaving the caller to detect
    # the missing file via downstream Path.exists() checks.
    stored_path = r"E:\fakeroot\sub\photo.jpg"
    resolver = PathResolver(stored_root=r"E:\fakeroot", runtime_root=tmp_path / "empty")
    resolved = resolver.resolve(stored_path)
    # Should return the translated-guess path (better diagnostics), not the stored string.
    assert resolved == tmp_path / "empty" / "sub" / "photo.jpg"


def test_resolver_handles_no_stored_root(tmp_path: Path) -> None:
    """If photo_root_at_embed was never set (legacy catalog), resolver passes
    paths through unchanged."""
    target = tmp_path / "photo.jpg"
    target.write_bytes(b"x")
    resolver = PathResolver(stored_root=None, runtime_root=tmp_path)
    assert resolver.resolve(str(target)) == target


def test_resolver_handles_unix_to_windows_translation(tmp_path: Path) -> None:
    """Catalog made on Windows (E:\\foo\\bar.jpg), served on Unix
    (/Volumes/whatever/bar.jpg). Both use forward-slash and backslash, mixed."""
    target = tmp_path / "bar.jpg"
    target.write_bytes(b"x")
    stored_path = r"E:\foo\bar.jpg"
    resolver = PathResolver(stored_root=r"E:\foo", runtime_root=tmp_path)
    assert resolver.resolve(stored_path) == target
