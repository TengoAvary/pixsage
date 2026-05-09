from __future__ import annotations

from PIL import Image

from pixsage.taggers.base import Tag, TagResult


class MockTagger:
    """Deterministic tagger for testing the orchestrator without loading real models."""

    def __init__(
        self,
        name: str,
        model_version: str,
        tags_per_call: list[tuple[str, float]],
        caption: str | None = None,
    ):
        self.name = name
        self.model_version = model_version
        self._tags = tags_per_call
        self._caption = caption
        self._loaded = False

    def load(self, device: str) -> None:
        self._loaded = True

    def tag(self, image: Image.Image) -> TagResult:
        if not self._loaded:
            raise RuntimeError("MockTagger.load() not called")
        tags = [Tag(name=n, confidence=c, hierarchy=None, source=self.name) for n, c in self._tags]
        return TagResult(tags=tags, caption=self._caption)
