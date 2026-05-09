from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pixsage.catalog import Catalog
from pixsage.cli import app
from pixsage.taggers.mock import MockTagger
from pixsage.xmp import read_xmp, write_xmp


EXIFTOOL = shutil.which("exiftool")
needs_exiftool = pytest.mark.skipif(EXIFTOOL is None, reason="exiftool not on PATH")

runner = CliRunner()


@pytest.fixture(autouse=True)
def use_mock_taggers(monkeypatch):
    def fake_build_taggers(_config):
        return [
            MockTagger(name="florence2", model_version="mock-1", tags_per_call=[("penguin", 1.0)], caption="A penguin."),
            MockTagger(name="ram++", model_version="mock-1", tags_per_call=[("bird", 0.9)]),
        ]
    monkeypatch.setattr("pixsage.cli.build_taggers", fake_build_taggers)


@needs_exiftool
def test_tag_writes_xmp_and_catalog(tmp_path: Path, make_jpeg):
    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    a = make_jpeg("a.jpg")
    a.rename(photo_root / "a.jpg")
    result = runner.invoke(app, ["tag", str(photo_root)])
    assert result.exit_code == 0, result.stdout
    fields = read_xmp(photo_root / "a.jpg", is_raw=False)
    assert "penguin" in fields.subject
    assert "bird" in fields.subject
    assert "auto-tagged-florence2" in fields.subject
    assert "auto-tagged-ram" in fields.subject
    assert fields.description == "A penguin."

    db = photo_root / ".photoindex" / "catalog.db"
    assert db.exists()
    cat = Catalog(db)
    cat.init_schema()
    runs = cat.list_runs()
    assert len(runs) == 1
    assert runs[0]["photos_processed"] == 1
    cat.close()


@needs_exiftool
def test_tag_skip_already_tagged(tmp_path: Path, make_jpeg):
    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    a = make_jpeg("a.jpg")
    a.rename(photo_root / "a.jpg")
    runner.invoke(app, ["tag", str(photo_root)])
    result = runner.invoke(app, ["tag", str(photo_root)])
    assert result.exit_code == 0
    cat = Catalog(photo_root / ".photoindex" / "catalog.db")
    cat.init_schema()
    runs = cat.list_runs()
    assert len(runs) == 2
    assert runs[1]["photos_processed"] == 0
    assert runs[1]["photos_skipped"] == 1
    cat.close()


@needs_exiftool
def test_force_retag(tmp_path: Path, make_jpeg):
    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    a = make_jpeg("a.jpg")
    a.rename(photo_root / "a.jpg")
    runner.invoke(app, ["tag", str(photo_root)])
    result = runner.invoke(app, ["tag", str(photo_root), "--force"])
    assert result.exit_code == 0
    cat = Catalog(photo_root / ".photoindex" / "catalog.db")
    cat.init_schema()
    runs = cat.list_runs()
    assert runs[1]["photos_processed"] == 1
    cat.close()


@needs_exiftool
def test_user_rejection_persists(tmp_path: Path, make_jpeg):
    """Remove an auto tag from XMP, --force re-run, expect tag stays removed."""
    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    a = make_jpeg("a.jpg")
    a.rename(photo_root / "a.jpg")
    runner.invoke(app, ["tag", str(photo_root)])
    # Remove "penguin" from XMP, leaving "bird".
    fields = read_xmp(photo_root / "a.jpg", is_raw=False)
    fields_minus = type(fields)(
        subject=[s for s in fields.subject if s != "penguin"],
        hierarchical_subject=fields.hierarchical_subject,
        description=fields.description,
    )
    write_xmp(photo_root / "a.jpg", fields_minus, is_raw=False)
    runner.invoke(app, ["tag", str(photo_root), "--force"])
    fields_after = read_xmp(photo_root / "a.jpg", is_raw=False)
    assert "penguin" not in fields_after.subject
    assert "bird" in fields_after.subject


@needs_exiftool
def test_sample_n(tmp_path: Path, make_jpeg):
    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    for i in range(5):
        p = make_jpeg(f"{i}.jpg")
        p.rename(photo_root / f"{i}.jpg")
    result = runner.invoke(app, ["tag", str(photo_root), "--sample", "2"])
    assert result.exit_code == 0
    cat = Catalog(photo_root / ".photoindex" / "catalog.db")
    cat.init_schema()
    runs = cat.list_runs()
    assert runs[0]["photos_processed"] == 2
    cat.close()


@needs_exiftool
def test_oom_retry_falls_back_to_smaller_size(tmp_path: Path, make_jpeg, monkeypatch):
    """Simulate OOM on first image-tag call; verify pipeline retries at smaller size."""
    photo_root = tmp_path / "photos"
    photo_root.mkdir()
    a = make_jpeg("a.jpg", size=(1500, 1000))
    a.rename(photo_root / "a.jpg")

    call_log: list[tuple[int, int]] = []

    class FlakyTagger:
        name = "florence2"
        model_version = "mock-1"
        def load(self, device): pass
        def tag(self, image):
            call_log.append(image.size)
            if len(call_log) == 1:
                raise RuntimeError("CUDA out of memory")
            from pixsage.taggers.base import Tag, TagResult
            return TagResult(tags=[Tag("ok", 1.0, None, "florence2")], caption=None)

    def fake_build(_cfg):
        return [FlakyTagger()]

    monkeypatch.setattr("pixsage.cli.build_taggers", fake_build)
    result = runner.invoke(app, ["tag", str(photo_root)])
    assert result.exit_code == 0, result.stdout
    assert len(call_log) >= 2
    assert max(call_log[1]) < max(call_log[0])  # second call used a smaller image
