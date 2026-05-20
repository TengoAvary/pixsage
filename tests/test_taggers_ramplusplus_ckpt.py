from __future__ import annotations

from pathlib import Path

from pixsage.taggers.ramplusplus import DEFAULT_CKPT_PATH, resolve_ram_ckpt


def test_resolve_uses_env_var_when_set(monkeypatch) -> None:
    monkeypatch.setenv("PIXSAGE_RAM_CKPT", "/custom/path/to/ckpt.pth")
    assert resolve_ram_ckpt() == "/custom/path/to/ckpt.pth"


def test_resolve_falls_back_to_default_when_env_var_unset(monkeypatch) -> None:
    monkeypatch.delenv("PIXSAGE_RAM_CKPT", raising=False)
    assert resolve_ram_ckpt() == str(DEFAULT_CKPT_PATH)


def test_resolve_falls_back_to_default_when_env_var_empty(monkeypatch) -> None:
    monkeypatch.setenv("PIXSAGE_RAM_CKPT", "")
    assert resolve_ram_ckpt() == str(DEFAULT_CKPT_PATH)


def test_default_path_under_home_cache_pixsage() -> None:
    assert DEFAULT_CKPT_PATH == Path.home() / ".cache" / "pixsage" / "ram_plus_swin_large_14m.pth"
