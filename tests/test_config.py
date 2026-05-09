from __future__ import annotations

from pathlib import Path

import pytest

from pixsage.config import Config, ensure_default_config, load_config


def test_load_config_minimal(tmp_path: Path):
    p = tmp_path / "vocabulary.toml"
    p.write_text("""
[florence2]
enabled = true
confidence_threshold = 0.5
exclude = []

[ram_plus_plus]
enabled = true
confidence_threshold = 0.4
exclude = []
""")
    cfg = load_config(p)
    assert isinstance(cfg, Config)
    assert cfg.florence2.enabled is True
    assert cfg.florence2.confidence_threshold == 0.5
    assert cfg.ram_plus_plus.confidence_threshold == 0.4


def test_load_config_with_hierarchy_overrides(tmp_path: Path):
    p = tmp_path / "vocabulary.toml"
    p.write_text("""
[florence2]
enabled = true
confidence_threshold = 0.5
exclude = ["x"]

[ram_plus_plus]
enabled = false
confidence_threshold = 0.4
exclude = []

[hierarchy_overrides]
"penguin" = "Wildlife|Bird|Penguin"
""")
    cfg = load_config(p)
    assert cfg.ram_plus_plus.enabled is False
    assert cfg.hierarchy_overrides == {"penguin": "Wildlife|Bird|Penguin"}


def test_load_config_invalid_threshold_raises(tmp_path: Path):
    p = tmp_path / "vocabulary.toml"
    p.write_text("""
[florence2]
enabled = true
confidence_threshold = "high"
exclude = []

[ram_plus_plus]
enabled = true
confidence_threshold = 0.5
exclude = []
""")
    with pytest.raises(ValueError):
        load_config(p)


def test_ensure_default_config_creates_file(tmp_path: Path):
    p = tmp_path / "vocabulary.toml"
    assert not p.exists()
    ensure_default_config(p)
    assert p.exists()
    cfg = load_config(p)
    assert cfg.florence2.enabled is True
    assert cfg.ram_plus_plus.enabled is True


def test_ensure_default_config_does_not_overwrite(tmp_path: Path):
    p = tmp_path / "vocabulary.toml"
    p.write_text('[florence2]\nenabled = false\nconfidence_threshold = 0.9\nexclude = []\n\n[ram_plus_plus]\nenabled = false\nconfidence_threshold = 0.9\nexclude = []\n')
    ensure_default_config(p)
    cfg = load_config(p)
    assert cfg.florence2.enabled is False  # untouched
