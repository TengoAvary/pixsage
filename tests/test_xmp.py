from __future__ import annotations

from pixsage.taggers.base import Tag, TagResult
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
