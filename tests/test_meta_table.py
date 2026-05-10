from pathlib import Path

import pytest

from pixsage.catalog import Catalog


def test_meta_set_and_get(tmp_path: Path) -> None:
    cat = Catalog(tmp_path / "test.db")
    cat.init_schema()
    cat.set_meta("photo_root_at_embed", r"E:\Sony alpha 7c")
    assert cat.get_meta("photo_root_at_embed") == r"E:\Sony alpha 7c"
    cat.close()


def test_meta_get_missing_returns_none(tmp_path: Path) -> None:
    cat = Catalog(tmp_path / "test.db")
    cat.init_schema()
    assert cat.get_meta("does_not_exist") is None
    cat.close()


def test_meta_overwrites_existing_key(tmp_path: Path) -> None:
    cat = Catalog(tmp_path / "test.db")
    cat.init_schema()
    cat.set_meta("k", "v1")
    cat.set_meta("k", "v2")
    assert cat.get_meta("k") == "v2"
    cat.close()
