"""Templates for the per-folder launcher files.

Two launchers per indexed folder:
- `Pixsage Search.bat` — Windows. Uses pythonw.exe (no console window).
- `Pixsage Search.command` — macOS. Terminal will flash briefly; tolerable for v1.

Both invoke `<runtime>/python -m pixsage serve <folder>` where <folder> is
the directory containing the launcher itself. `pixsage serve` opens the
default browser on its own (cli.py:609-611).
"""
from __future__ import annotations


WINDOWS_BAT = r"""@echo off
REM Pixsage Search launcher (Windows).
REM Runs the locally-installed pixsage runtime against this folder.
REM PYTHONNOUSERSITE=1 isolates the runtime from any host-Python user site-packages
REM (e.g. a torch nightly the user installed for unrelated work) — without it,
REM the host's torchvision can leak in and crash against the runtime's torch.
set PYTHONNOUSERSITE=1
start "" "{runtime_path}\python\pythonw.exe" -m pixsage serve "%~dp0"
"""


MACOS_COMMAND = r"""#!/bin/bash
# Pixsage Search launcher (macOS).
# Runs the locally-installed pixsage runtime against this folder.
# PYTHONNOUSERSITE=1 isolates the runtime from any host-Python user site-packages.
cd "$(dirname "$0")"
export PYTHONNOUSERSITE=1
exec "{runtime_path}/python/bin/python3" -m pixsage serve "$PWD"
"""


def render(template: str, runtime_path: str) -> str:
    """Substitute {runtime_path} into a template. No other placeholders."""
    return template.replace("{runtime_path}", runtime_path)
