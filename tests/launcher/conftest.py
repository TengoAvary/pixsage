from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _sandbox_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect Path.home() (and HOME/USERPROFILE) to a temp dir for every
    launcher test.

    ``install_runtime._install_laptop_launcher`` writes the laptop launcher to
    ``Path.home()/Applications/Pixsage Search.command`` (macOS) or
    ``Path.home()/Desktop/Pixsage Search.bat`` (Windows) using a *hardcoded*
    real home — and it branches on the live ``sys.platform``, not the build
    target. Any launcher test that reaches ``install_runtime_via_build``
    without redirecting home would therefore overwrite the developer's real
    installed launcher (observed: a pytest tmp path ending up inside
    ~/Applications/Pixsage Search.command). This autouse fixture makes the
    whole launcher suite home-safe so that can't happen again.
    """
    # Distinct name so it never collides with a test that creates its own
    # tmp_path/"home" (e.g. test_install_runtime_drops_laptop_launcher_on_macos,
    # which sets up and monkeypatches its own home — that override wins for
    # that test, which is fine; both are under tmp_path).
    fake_home = tmp_path / "_pytest_sandbox_home"
    fake_home.mkdir()
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: fake_home))
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    return fake_home
