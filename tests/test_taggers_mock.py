from __future__ import annotations

from PIL import Image

from pixsage.taggers.mock import MockTagger


def test_mock_tagger_returns_configured_tags():
    tagger = MockTagger(
        name="florence2",
        model_version="mock-1",
        tags_per_call=[("penguin", 1.0), ("ice", 0.9)],
        caption="A penguin on ice.",
    )
    tagger.load("cpu")
    img = Image.new("RGB", (10, 10))
    result = tagger.tag(img)
    assert {t.name for t in result.tags} == {"penguin", "ice"}
    assert all(t.source == "florence2" for t in result.tags)
    assert result.caption == "A penguin on ice."


def test_mock_tagger_no_caption():
    tagger = MockTagger(name="ram++", model_version="mock-1", tags_per_call=[("bird", 0.8)])
    tagger.load("cpu")
    result = tagger.tag(Image.new("RGB", (10, 10)))
    assert result.caption is None
    assert result.tags[0].source == "ram++"
