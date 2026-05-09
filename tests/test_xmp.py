from __future__ import annotations

from pixsage.taggers.base import Tag
from pixsage.xmp import XmpFields, merge_xmp


def test_merge_adds_new_auto_tags():
    existing = XmpFields(subject=["antarctica"], hierarchical_subject=[], description=None)
    new = [
        Tag("penguin", 1.0, "Wildlife|Bird|Penguin", "florence2"),
    ]
    merged = merge_xmp(
        existing=existing,
        new_tags=new,
        previously_applied={("penguin", "florence2")},  # already in our DB
        user_rejected=set(),
        caption="A penguin.",
        caption_overwrite=False,
        sources_with_tags={"florence2"},
    )
    assert "penguin" in merged.subject
    assert "antarctica" in merged.subject
    assert "auto-tagged-florence2" in merged.subject
    assert "Wildlife|Bird|Penguin" in merged.hierarchical_subject
    assert merged.description == "A penguin."


def test_merge_preserves_user_keywords():
    existing = XmpFields(subject=["my keyword", "another"], hierarchical_subject=[], description=None)
    merged = merge_xmp(
        existing=existing,
        new_tags=[Tag("penguin", 1.0, None, "florence2")],
        previously_applied=set(),
        user_rejected=set(),
        caption=None,
        caption_overwrite=False,
        sources_with_tags={"florence2"},
    )
    assert "my keyword" in merged.subject
    assert "another" in merged.subject
    assert "penguin" in merged.subject


def test_merge_skips_user_rejected_tags():
    existing = XmpFields(subject=[], hierarchical_subject=[], description=None)
    merged = merge_xmp(
        existing=existing,
        new_tags=[
            Tag("penguin", 1.0, None, "florence2"),
            Tag("ice", 0.9, None, "florence2"),
        ],
        previously_applied={("penguin", "florence2"), ("ice", "florence2")},
        user_rejected={("ice", "florence2")},
        caption=None,
        caption_overwrite=False,
        sources_with_tags={"florence2"},
    )
    assert "penguin" in merged.subject
    assert "ice" not in merged.subject


def test_merge_does_not_overwrite_existing_description():
    existing = XmpFields(subject=[], hierarchical_subject=[], description="Photographer's caption")
    merged = merge_xmp(
        existing=existing,
        new_tags=[Tag("penguin", 1.0, None, "florence2")],
        previously_applied=set(),
        user_rejected=set(),
        caption="Auto caption",
        caption_overwrite=False,
        sources_with_tags={"florence2"},
    )
    assert merged.description == "Photographer's caption"


def test_merge_overwrites_when_configured():
    existing = XmpFields(subject=[], hierarchical_subject=[], description="old")
    merged = merge_xmp(
        existing=existing,
        new_tags=[],
        previously_applied=set(),
        user_rejected=set(),
        caption="new",
        caption_overwrite=True,
        sources_with_tags=set(),
    )
    assert merged.description == "new"


def test_merge_marker_tags_per_source():
    existing = XmpFields(subject=[], hierarchical_subject=[], description=None)
    merged = merge_xmp(
        existing=existing,
        new_tags=[
            Tag("penguin", 1.0, None, "florence2"),
            Tag("bird", 0.9, None, "ram++"),
        ],
        previously_applied=set(),
        user_rejected=set(),
        caption=None,
        caption_overwrite=False,
        sources_with_tags={"florence2", "ram++"},
    )
    assert "auto-tagged-florence2" in merged.subject
    assert "auto-tagged-ram" in merged.subject


def test_merge_no_marker_tag_when_source_has_no_new_tags():
    existing = XmpFields(subject=[], hierarchical_subject=[], description=None)
    merged = merge_xmp(
        existing=existing,
        new_tags=[Tag("penguin", 1.0, None, "florence2")],
        previously_applied=set(),
        user_rejected=set(),
        caption=None,
        caption_overwrite=False,
        sources_with_tags={"florence2"},
    )
    assert "auto-tagged-ram" not in merged.subject


import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from pixsage.xmp import read_xmp, write_xmp  # noqa: E402

EXIFTOOL = shutil.which("exiftool")
needs_exiftool = pytest.mark.skipif(EXIFTOOL is None, reason="exiftool not on PATH")


@needs_exiftool
def test_write_and_read_jpeg(make_jpeg):
    p = make_jpeg("rt.jpg")
    fields = XmpFields(
        subject=["penguin", "ice"],
        hierarchical_subject=["Wildlife|Bird|Penguin"],
        description="A penguin on ice.",
    )
    write_xmp(p, fields, is_raw=False)
    got = read_xmp(p, is_raw=False)
    assert set(got.subject) >= {"penguin", "ice"}
    assert "Wildlife|Bird|Penguin" in got.hierarchical_subject
    assert got.description == "A penguin on ice."


@needs_exiftool
def test_write_raw_uses_sidecar(tmp_path: Path):
    # We don't need a real raw — exiftool will create a sidecar even from a fake path
    # as long as we tell it to write to <path>.xmp explicitly.
    fake_raw = tmp_path / "fake.arw"
    fake_raw.write_bytes(b"\x00")  # contents irrelevant; exiftool only reads/writes the sidecar
    fields = XmpFields(subject=["penguin"], hierarchical_subject=[], description=None)
    write_xmp(fake_raw, fields, is_raw=True)
    sidecar = tmp_path / "fake.xmp"
    assert sidecar.exists()
    got = read_xmp(fake_raw, is_raw=True)
    assert "penguin" in got.subject


@needs_exiftool
def test_read_xmp_returns_empty_when_no_sidecar(tmp_path: Path):
    p = tmp_path / "no_sidecar.arw"
    p.write_bytes(b"\x00")
    fields = read_xmp(p, is_raw=True)
    assert fields.subject == []
    assert fields.hierarchical_subject == []
    assert fields.description is None
