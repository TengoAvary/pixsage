from pathlib import Path
from unittest.mock import patch

import pytest


def test_install_paths_match_canonical(tmp_path: Path, monkeypatch) -> None:
    from scripts.launcher.install_runtime import canonical_install_path

    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    win_path = canonical_install_path()
    assert win_path == tmp_path / "local" / "pixsage"

    monkeypatch.setattr("sys.platform", "darwin")
    mac_path = canonical_install_path(home_override=tmp_path)
    assert mac_path == tmp_path / "Library" / "Application Support" / "pixsage"


def test_install_runtime_calls_build_and_download(tmp_path: Path) -> None:
    from scripts.launcher.install_runtime import install_runtime_via_build

    captured = {}

    def fake_build(target_name, out_dir, **kwargs):
        captured["build_target"] = target_name
        captured["build_out"] = out_dir
        return out_dir / "python" / "python.exe"

    def fake_download(out_dir):
        captured["download_out"] = out_dir

    with patch("scripts.launcher.install_runtime.build_runtime", side_effect=fake_build), \
         patch("scripts.launcher.install_runtime.download_models", side_effect=fake_download):
        install_runtime_via_build(install_dir=tmp_path / "install", target="windows-x64")

    assert captured["build_target"] == "windows-x64"
    assert captured["build_out"] == tmp_path / "install"
    assert captured["download_out"] == tmp_path / "install"


def test_install_runtime_skips_when_already_present(tmp_path: Path) -> None:
    """If the install dir already has python/, install does nothing unless --force."""
    from scripts.launcher.install_runtime import install_runtime_via_build

    install_dir = tmp_path / "install"
    (install_dir / "python").mkdir(parents=True)

    with patch("scripts.launcher.install_runtime.build_runtime") as build, \
         patch("scripts.launcher.install_runtime.download_models") as dl:
        install_runtime_via_build(install_dir=install_dir, target="windows-x64", force=False)

    build.assert_not_called()
    dl.assert_not_called()
