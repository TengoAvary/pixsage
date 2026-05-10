import stat
from pathlib import Path

import pytest


def test_stage_folder_writes_both_launchers(tmp_path: Path) -> None:
    from scripts.launcher.stage_folder import stage_folder

    folder = tmp_path / "Sony alpha 7c"
    folder.mkdir()
    stage_folder(folder, runtime_path_windows=r"%LOCALAPPDATA%\pixsage",
                 runtime_path_macos="$HOME/Library/Application Support/pixsage")

    bat = folder / "Pixsage Search.bat"
    cmd = folder / "Pixsage Search.command"
    assert bat.exists()
    assert cmd.exists()

    bat_body = bat.read_text(encoding="utf-8")
    assert "%LOCALAPPDATA%\\pixsage\\python\\pythonw.exe" in bat_body
    assert "-m pixsage serve" in bat_body

    cmd_body = cmd.read_text(encoding="utf-8")
    assert "$HOME/Library/Application Support/pixsage/python/bin/python3" in cmd_body


def test_stage_folder_makes_command_executable(tmp_path: Path) -> None:
    """The .command file must have +x for macOS to launch it on double-click."""
    from scripts.launcher.stage_folder import stage_folder

    folder = tmp_path / "shoot"
    folder.mkdir()
    stage_folder(folder)

    cmd = folder / "Pixsage Search.command"
    mode = cmd.stat().st_mode
    import sys
    if sys.platform != "win32":
        assert mode & stat.S_IXUSR, f"missing +x on {cmd}"


def test_stage_folder_idempotent(tmp_path: Path) -> None:
    from scripts.launcher.stage_folder import stage_folder

    folder = tmp_path / "shoot"
    folder.mkdir()
    stage_folder(folder)
    stage_folder(folder)
    assert (folder / "Pixsage Search.bat").exists()
    assert (folder / "Pixsage Search.command").exists()


def test_stage_folder_rejects_missing_dir(tmp_path: Path) -> None:
    from scripts.launcher.stage_folder import stage_folder

    nope = tmp_path / "does-not-exist"
    with pytest.raises((FileNotFoundError, NotADirectoryError)):
        stage_folder(nope)
