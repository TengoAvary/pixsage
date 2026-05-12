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


def test_install_runtime_calls_build_then_downloads_via_runtime_python(tmp_path: Path) -> None:
    """install_runtime should (1) call build_runtime, then (2) subprocess the
    runtime python to download models — so the bootstrap python doesn't need
    huggingface_hub installed."""
    from scripts.launcher.install_runtime import install_runtime_via_build

    captured: dict = {}

    def fake_build(target_name, out_dir, **kwargs):
        captured["build_target"] = target_name
        captured["build_out"] = out_dir
        # Mimic the real layout so install_runtime can locate python_exe.
        (out_dir / "python").mkdir(parents=True, exist_ok=True)
        return out_dir / "python" / "python.exe"

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env", {})
        class R:
            returncode = 0
        return R()

    with patch("scripts.launcher.install_runtime.build_runtime", side_effect=fake_build), \
         patch("scripts.launcher.install_runtime.subprocess.run", side_effect=fake_run):
        install_runtime_via_build(install_dir=tmp_path / "install", target="windows-x64")

    assert captured["build_target"] == "windows-x64"
    assert captured["build_out"] == tmp_path / "install"
    cmd = captured["cmd"]
    # The runtime python (not sys.executable) is invoked
    assert "windows-x64" in str(cmd[0]) or cmd[0].endswith(("python.exe", "python3"))
    assert cmd[1:5] == ["-m", "scripts.launcher.download_models", "--out", str(tmp_path / "install")]
    env = captured["env"]
    assert "PYTHONPATH" in env
    assert str(tmp_path / "install" / "site-packages") in env["PYTHONPATH"]
    assert env.get("PYTHONNOUSERSITE") == "1"


def test_install_runtime_skips_when_already_present(tmp_path: Path) -> None:
    """If the install dir already has python/, install does nothing unless --force."""
    from scripts.launcher.install_runtime import install_runtime_via_build

    install_dir = tmp_path / "install"
    (install_dir / "python").mkdir(parents=True)

    with patch("scripts.launcher.install_runtime.build_runtime") as build, \
         patch("scripts.launcher.install_runtime.subprocess.run") as run:
        install_runtime_via_build(install_dir=install_dir, target="windows-x64", force=False)

    build.assert_not_called()
    run.assert_not_called()
