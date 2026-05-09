from __future__ import annotations

from pixsage.config import Config, TaggerConfig
from pixsage.taggers.base import Tag


def filter_tags(tags: list[Tag], config: Config) -> list[Tag]:
    out: list[Tag] = []
    overrides = {k.lower(): v for k, v in config.hierarchy_overrides.items()}
    for tag in tags:
        cfg = _config_for_source(tag.source, config)
        if cfg is None or not cfg.enabled or not cfg.tags_enabled:
            continue
        if tag.confidence < cfg.confidence_threshold:
            continue
        if any(tag.name.lower() == ex.lower() for ex in cfg.exclude):
            continue
        hierarchy = overrides.get(tag.name.lower(), tag.hierarchy)
        out.append(Tag(name=tag.name, confidence=tag.confidence, hierarchy=hierarchy, source=tag.source))
    return out


def _config_for_source(source: str, config: Config) -> TaggerConfig | None:
    if source == "florence2":
        return config.florence2
    if source == "ram++":
        return config.ram_plus_plus
    return None
