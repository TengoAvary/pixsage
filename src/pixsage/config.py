from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


class TaggerConfig(BaseModel):
    enabled: bool = True
    # When `enabled = true` and `tags_enabled = false`, the tagger still runs —
    # any caption it produces still flows through to dc:description — but its
    # tags are dropped before merge. Lets you, e.g., keep Florence-2's caption
    # while using only RAM++ as a tag source.
    tags_enabled: bool = True
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    exclude: list[str] = Field(default_factory=list)


class CaptionConfig(BaseModel):
    enabled: bool = True
    overwrite: bool = False


class SigLIP2Config(BaseModel):
    enabled: bool = True
    model: str = "google/siglip2-so400m-patch14-384"
    image: bool = True
    caption: bool = True
    batch_size: int = 16


class EmbeddingsConfig(BaseModel):
    enabled: bool = True
    siglip2: SigLIP2Config = Field(default_factory=SigLIP2Config)


class SearchConfig(BaseModel):
    default_image_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    top_k: int = 60
    thumb_size_default: str = "medium"  # "small" | "medium" | "large"


class Config(BaseModel):
    florence2: TaggerConfig
    ram_plus_plus: TaggerConfig
    hierarchy_overrides: dict[str, str] = Field(default_factory=dict)
    caption: CaptionConfig = Field(default_factory=CaptionConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)


DEFAULT_CONFIG_TOML = """\
# pixsage vocabulary configuration. Edit and re-run `pixsage tag --force` to apply.

# Florence-2 produces good captions but its region/object outputs as
# *tags* tend to be multi-word region descriptions ("traditional Dutch
# houses along canal in Bruges, Belgium") that don't compose with
# Lightroom's exact-match keyword filtering. RAM++ is the cleaner tag
# source for keywords. Default keeps Florence-2 as caption-only.
# Set florence2.tags_enabled = true if you want the region phrases too.
[florence2]
enabled = true
tags_enabled = false
confidence_threshold = 0.5
exclude = ["photograph", "image", "picture"]

[ram_plus_plus]
enabled = true
tags_enabled = true
confidence_threshold = 0.4
exclude = []

[hierarchy_overrides]
# flat tag (lowercase) = "Top|Mid|Leaf"
# example:
# "penguin" = "Wildlife|Bird|Penguin"

[caption]
enabled = true
overwrite = false

[embeddings]
enabled = true

[embeddings.siglip2]
enabled = true
model = "google/siglip2-so400m-patch14-384"
image = true
caption = true
batch_size = 16

[search]
default_image_weight = 0.5
top_k = 60
thumb_size_default = "medium"
"""


def load_config(path: Path) -> Config:
    with path.open("rb") as f:
        data = tomllib.load(f)
    try:
        return Config.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Invalid config at {path}: {e}") from e


def ensure_default_config(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
