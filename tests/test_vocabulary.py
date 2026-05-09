from __future__ import annotations

from pixsage.config import CaptionConfig, Config, TaggerConfig
from pixsage.taggers.base import Tag
from pixsage.vocabulary import filter_tags


def make_config(
    fl_enabled=True, fl_threshold=0.5, fl_exclude=None,
    ram_enabled=True, ram_threshold=0.4, ram_exclude=None,
    hierarchy_overrides=None,
):
    return Config(
        florence2=TaggerConfig(enabled=fl_enabled, confidence_threshold=fl_threshold, exclude=fl_exclude or []),
        ram_plus_plus=TaggerConfig(enabled=ram_enabled, confidence_threshold=ram_threshold, exclude=ram_exclude or []),
        hierarchy_overrides=hierarchy_overrides or {},
        caption=CaptionConfig(),
    )


def test_filter_drops_below_threshold():
    cfg = make_config(fl_threshold=0.6)
    tags = [
        Tag(name="penguin", confidence=0.7, hierarchy=None, source="florence2"),
        Tag(name="ice", confidence=0.5, hierarchy=None, source="florence2"),
    ]
    out = filter_tags(tags, cfg)
    assert [t.name for t in out] == ["penguin"]


def test_filter_drops_excluded_case_insensitive():
    cfg = make_config(fl_exclude=["Photograph"])
    tags = [
        Tag(name="photograph", confidence=1.0, hierarchy=None, source="florence2"),
        Tag(name="penguin", confidence=1.0, hierarchy=None, source="florence2"),
    ]
    out = filter_tags(tags, cfg)
    assert [t.name for t in out] == ["penguin"]


def test_filter_disables_tagger():
    cfg = make_config(ram_enabled=False)
    tags = [
        Tag(name="penguin", confidence=1.0, hierarchy=None, source="florence2"),
        Tag(name="bird", confidence=1.0, hierarchy=None, source="ram++"),
    ]
    out = filter_tags(tags, cfg)
    assert [t.source for t in out] == ["florence2"]


def test_filter_applies_hierarchy_override():
    cfg = make_config(hierarchy_overrides={"penguin": "Wildlife|Bird|Penguin"})
    tags = [
        Tag(name="Penguin", confidence=1.0, hierarchy=None, source="florence2"),
    ]
    out = filter_tags(tags, cfg)
    assert out[0].hierarchy == "Wildlife|Bird|Penguin"


def test_filter_preserves_existing_hierarchy_when_no_override():
    cfg = make_config()
    tags = [
        Tag(name="penguin", confidence=1.0, hierarchy="Wildlife|Bird|Penguin", source="florence2"),
    ]
    out = filter_tags(tags, cfg)
    assert out[0].hierarchy == "Wildlife|Bird|Penguin"
