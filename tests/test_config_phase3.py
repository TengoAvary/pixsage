from __future__ import annotations

from pathlib import Path

from pixsage.config import DEFAULT_CONFIG_TOML, ensure_default_config, load_config


def test_default_toml_includes_embeddings_block(tmp_path: Path):
    cfg_path = tmp_path / "vocabulary.toml"
    ensure_default_config(cfg_path)
    text = cfg_path.read_text(encoding="utf-8")
    assert "[embeddings]" in text
    assert "[embeddings.siglip2]" in text
    assert "[search]" in text


def test_loaded_config_has_embeddings_defaults(tmp_path: Path):
    cfg_path = tmp_path / "vocabulary.toml"
    ensure_default_config(cfg_path)
    cfg = load_config(cfg_path)

    assert cfg.embeddings.enabled is True
    assert cfg.embeddings.siglip2.enabled is True
    assert cfg.embeddings.siglip2.image is True
    assert cfg.embeddings.siglip2.caption is True
    assert cfg.embeddings.siglip2.batch_size == 16


def test_search_config_defaults(tmp_path: Path):
    cfg_path = tmp_path / "vocabulary.toml"
    ensure_default_config(cfg_path)
    cfg = load_config(cfg_path)

    assert cfg.search.default_image_weight == 0.5
    assert cfg.search.top_k == 60
    assert cfg.search.thumb_size_default == "medium"
