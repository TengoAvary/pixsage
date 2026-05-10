"""Drop Pixsage Search launchers into an indexed folder.

Idempotent: re-running overwrites existing launchers (in case the runtime
path or template changed between versions).
"""
from __future__ import annotations

import argparse
import stat
import sys
from pathlib import Path

from scripts.launcher.launcher_templates import (
    MACOS_COMMAND,
    WINDOWS_BAT,
    render,
)


DEFAULT_RUNTIME_WINDOWS = r"%LOCALAPPDATA%\pixsage"
DEFAULT_RUNTIME_MACOS = "$HOME/Library/Application Support/pixsage"


def stage_folder(
    folder: Path,
    runtime_path_windows: str = DEFAULT_RUNTIME_WINDOWS,
    runtime_path_macos: str = DEFAULT_RUNTIME_MACOS,
) -> None:
    """Write `Pixsage Search.bat` + `Pixsage Search.command` into `folder`."""
    if not folder.is_dir():
        raise FileNotFoundError(f"not a directory: {folder}")

    bat = folder / "Pixsage Search.bat"
    cmd = folder / "Pixsage Search.command"

    bat.write_text(render(WINDOWS_BAT, runtime_path=runtime_path_windows), encoding="utf-8")
    cmd.write_text(render(MACOS_COMMAND, runtime_path=runtime_path_macos), encoding="utf-8")

    cmd_mode = cmd.stat().st_mode
    cmd.chmod(cmd_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage Pixsage Search launchers in a folder.")
    parser.add_argument("folder", type=Path, help="The indexed folder (must contain .photoindex/).")
    parser.add_argument("--runtime-windows", default=DEFAULT_RUNTIME_WINDOWS)
    parser.add_argument("--runtime-macos", default=DEFAULT_RUNTIME_MACOS)
    args = parser.parse_args()

    stage_folder(args.folder, args.runtime_windows, args.runtime_macos)
    print(f"Staged launchers in: {args.folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
