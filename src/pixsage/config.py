from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


class TaggerConfig(BaseModel):
    enabled: bool = True
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    exclude: list[str] = Field(default_factory=list)


class CaptionConfig(BaseModel):
    enabled: bool = True
    overwrite: bool = False


class Config(BaseModel):
    florence2: TaggerConfig
    ram_plus_plus: TaggerConfig
    hierarchy_overrides: dict[str, str] = Field(default_factory=dict)
    caption: CaptionConfig = Field(default_factory=CaptionConfig)


DEFAULT_CONFIG_TOML = """\
# pixsage vocabulary configuration. Edit and re-run `pixsage tag --force` to apply.

[florence2]
enabled = true
confidence_threshold = 0.5
exclude = ["photograph", "image", "picture"]

[ram_plus_plus]
enabled = true
confidence_threshold = 0.4
exclude = []

[hierarchy_overrides]
# flat tag (lowercase) = "Top|Mid|Leaf"
# example:
# "penguin" = "Wildlife|Bird|Penguin"

[caption]
enabled = true
overwrite = false
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
