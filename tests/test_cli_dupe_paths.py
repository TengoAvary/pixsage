"""Tests for the duplicate-path handling in `pixsage tag`.

The photographer's corpus has byte-identical files at multiple paths (e.g.
`Seymour/DSC01730.ARW` and `Seymour/Fieldwork/K-Pg/DSC01730.ARW` for a
two-axis date+topic organization). All copies need their own XMP sidecar
so Lightroom shows the keywords whichever folder is imported.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pixsage.catalog import Catalog
from pixsage.cli import app
from pixsage.taggers.mock import MockTagger
from pixsage.xmp import read_xmp


EXIFTOOL = shutil.which("exiftool")
needs_exiftool = pytest.mark.skipif(EXIFTOOL is None, reason="exiftool not on PATH")

runner = CliRunner()


_TEST_CONFIG_TOML = """\
[florence2]
enabled = true
tags_enabled = true
confidence_threshold = 0.5
exclude = []

[ram_plus_plus]
enabled = true
tags_enabled = true
confidence_threshold = 0.4
exclude = []

[caption]
enabled = true
overwrite = false
"""


class CountingMockTagger(MockTagger):
    """Mock tagger that records how many times tag() is called."""
    def __init__(self, *args, calls: list, **kwargs):
        super().__init__(*args, **kwargs)
        self._calls = calls

    def tag(self, image):
        self._calls.append(1)
        return super().tag(image)


@pytest.fixture(autouse=True)
def use_mock_taggers(monkeypatch):
    monkeypatch.setattr("pixsage.config.DEFAULT_CONFIG_TOML", _TEST_CONFIG_TOML)


def _install_counting_taggers(monkeypatch) -> list:
    calls: list = []
    def fake_build(_cfg):
        return [
            CountingMockTagger(
                name="florence2", model_version="mock-1",
                tags_per_call=[("penguin", 1.0)], caption="A penguin.",
                calls=calls,
            ),
            CountingMockTagger(
                name="ram++", model_version="mock-1",
                tags_per_call=[("bird", 0.9)],
                calls=calls,
            ),
        ]
    monkeypatch.setattr("pixsage.cli.build_taggers", fake_build)
    return calls


def _make_dupe_set(make_jpeg, photo_root: Path, count: int = 3) -> list[Path]:
    """Create `count` byte-identical JPGs at different paths under photo_root."""
    primary = make_jpeg("dupe.jpg")
    paths = []
    for i in range(count):
        sub = photo_root / f"folder_{i}"
        sub.mkdir(parents=True, exist_ok=True)
        target = sub / "dupe.jpg"
        shutil.copyfile(primary, target)
        paths.append(target)
    primary.unlink()
    return paths


@needs_exiftool
def test_dupe_paths_in_same_run_all_get_xmp(tmp_path, make_jpeg, monkeypatch):
    """Three byte-identical JPGs in three folders → all three get XMP sidecars."""
    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    paths = _make_dupe_set(make_jpeg, photo_root, count=3)
    calls = _install_counting_taggers(monkeypatch)

    result = runner.invoke(app, ["tag", str(photo_root)])
    assert result.exit_code == 0, result.stdout

    # All three paths have the same auto-tags + caption.
    for p in paths:
        # Path may have rekeyed (embedded XMP changes JPEG bytes) — read from disk.
        fields = read_xmp(p, is_raw=False)
        assert "penguin" in fields.subject
        assert "bird" in fields.subject
        assert fields.description == "A penguin."


@needs_exiftool
def test_dupe_paths_run_model_only_once(tmp_path, make_jpeg, monkeypatch):
    """The cache means model.tag() runs once per sha, even with N dupe paths."""
    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    _make_dupe_set(make_jpeg, photo_root, count=4)
    calls = _install_counting_taggers(monkeypatch)

    result = runner.invoke(app, ["tag", str(photo_root)])
    assert result.exit_code == 0, result.stdout

    # 2 taggers × 1 unique sha = 2 model calls (not 8).
    assert len(calls) == 2, f"expected 2 model calls, got {len(calls)}"


@needs_exiftool
def test_dupe_paths_reported_in_summary(tmp_path, make_jpeg, monkeypatch):
    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    _make_dupe_set(make_jpeg, photo_root, count=3)
    _install_counting_taggers(monkeypatch)

    result = runner.invoke(app, ["tag", str(photo_root)])
    assert result.exit_code == 0
    assert "processed=3" in result.stdout
    assert "model_runs=1" in result.stdout
    assert "dupe_writes=2" in result.stdout


@needs_exiftool
def test_dupe_path_added_after_first_run_gets_xmp_via_reconstitution(
    tmp_path, make_jpeg, monkeypatch
):
    """First run tags one path. Then a dupe path is added. Re-run reconstitutes
    tags from the catalog and writes XMP to the new path without re-running the
    model.
    """
    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    primary_dir = photo_root / "primary"
    primary_dir.mkdir()
    src = make_jpeg("dupe.jpg")
    primary = primary_dir / "dupe.jpg"
    shutil.copyfile(src, primary)
    src.unlink()

    calls = _install_counting_taggers(monkeypatch)
    runner.invoke(app, ["tag", str(photo_root)])
    assert len(calls) == 2  # one model run for the one photo

    # Add a dupe path
    dupe_dir = photo_root / "dupe_dir"
    dupe_dir.mkdir()
    dupe = dupe_dir / "dupe.jpg"
    # Use the post-tag content (sha may have rekeyed)
    shutil.copyfile(primary, dupe)

    calls.clear()
    result = runner.invoke(app, ["tag", str(photo_root)])
    assert result.exit_code == 0

    # Model didn't run again — the cache was reconstituted from the catalog.
    assert len(calls) == 0, f"expected 0 model calls on re-run, got {len(calls)}"

    # Both paths now have XMP
    for p in (primary, dupe):
        fields = read_xmp(p, is_raw=False)
        assert "penguin" in fields.subject
        assert fields.description == "A penguin."


@needs_exiftool
def test_single_path_rerun_still_skips(tmp_path, make_jpeg, monkeypatch):
    """Non-dupe path on a re-run should still skip (preserves existing semantics)."""
    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    src = make_jpeg("solo.jpg")
    shutil.copyfile(src, photo_root / "solo.jpg")
    src.unlink()

    _install_counting_taggers(monkeypatch)
    runner.invoke(app, ["tag", str(photo_root)])
    result = runner.invoke(app, ["tag", str(photo_root)])

    assert result.exit_code == 0
    assert "processed=0" in result.stdout
    assert "skipped=1" in result.stdout


@needs_exiftool
def test_rewrite_with_dupe_paths_strips_prior_tags_from_each(
    tmp_path, make_jpeg, monkeypatch
):
    """--rewrite + dupe paths: every path's XMP loses the v1 auto-tags before
    the v2 tags are applied. Pre-existing user keywords on each path survive
    independently.
    """
    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    paths = _make_dupe_set(make_jpeg, photo_root, count=2)

    # v1 run
    _install_counting_taggers(monkeypatch)
    runner.invoke(app, ["tag", str(photo_root)])
    for p in paths:
        f = read_xmp(p, is_raw=False)
        assert "penguin" in f.subject

    # Switch to v2, --rewrite
    calls_v2: list = []
    def fake_v2(_cfg):
        return [
            CountingMockTagger(
                name="florence2", model_version="mock-2",
                tags_per_call=[("ice", 1.0)], caption="Just ice.",
                calls=calls_v2,
            ),
            CountingMockTagger(
                name="ram++", model_version="mock-2",
                tags_per_call=[("cold", 0.9)],
                calls=calls_v2,
            ),
        ]
    monkeypatch.setattr("pixsage.cli.build_taggers", fake_v2)
    result = runner.invoke(app, ["tag", str(photo_root), "--rewrite"])
    assert result.exit_code == 0, result.stdout

    for p in paths:
        f = read_xmp(p, is_raw=False)
        assert "penguin" not in f.subject, f"prior auto-tag survived rewrite at {p}"
        assert "bird" not in f.subject
        assert "ice" in f.subject
        assert "cold" in f.subject
        assert f.description == "Just ice."

    # Model ran once for the unique sha, not twice
    assert len(calls_v2) == 2  # 2 taggers × 1 sha


@needs_exiftool
def test_dupe_paths_distinct_user_keywords_preserved(
    tmp_path, make_jpeg, monkeypatch
):
    """If two dupe paths have different user keywords (same content but the
    photographer keyworded them in Lightroom independently), each path keeps
    its own user keywords plus the auto-tags.
    """
    from pixsage.xmp import XmpFields, write_xmp

    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    paths = _make_dupe_set(make_jpeg, photo_root, count=2)

    write_xmp(paths[0], XmpFields(subject=["antarctica"], hierarchical_subject=[], description=None), is_raw=False)
    write_xmp(paths[1], XmpFields(subject=["fieldwork"], hierarchical_subject=[], description=None), is_raw=False)

    _install_counting_taggers(monkeypatch)
    result = runner.invoke(app, ["tag", str(photo_root)])
    assert result.exit_code == 0, result.stdout

    f0 = read_xmp(paths[0], is_raw=False)
    f1 = read_xmp(paths[1], is_raw=False)
    assert "antarctica" in f0.subject
    assert "fieldwork" in f1.subject
    # Both have the auto-tags
    for f in (f0, f1):
        assert "penguin" in f.subject
        assert "bird" in f.subject
