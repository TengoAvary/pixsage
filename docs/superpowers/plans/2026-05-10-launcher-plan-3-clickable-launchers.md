# Launcher Plan 3: Clickable Per-Folder Launchers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute task-by-task.

**Goal:** Photographer double-clicks `Pixsage Search` in any indexed folder and the search webapp opens in her browser. Minimum viable: shell-script launchers (`.bat` on Windows, `.command` on Mac) that invoke a locally-installed pixsage runtime. No native binary, no tray icon, no first-launch installer dialog — those are deferred polish.

**Architecture:**
- One-time setup on her machine: extract a pre-built runtime tarball (Plan 2's output) into `%LOCALAPPDATA%\pixsage\` (Windows) or `~/Library/Application Support/pixsage/` (Mac). A new `scripts/launcher/install_runtime.py` does this from a tarball OR by running build_runtime + download_models live.
- Per-folder: each indexed folder gets `Pixsage Search.bat` and `Pixsage Search.command` (~5 lines each). Both invoke `<runtime>/python -m pixsage serve <containing-folder>`. `stage_folder.py` drops these in.
- The `serve` command already opens the browser via its existing webbrowser hook (line 611 of cli.py). No new launcher logic needed in pixsage core.

**Tech Stack:** Python (staging scripts), shell scripting (the launcher templates).

**Companion plans:**
- Plan 1 (path translation): shipped. Handles drive-letter / mount-point differences.
- Plan 2 (runtime build): shipped. Produces `_PixsageRuntime/` trees + pre-staged HF cache.

**Deferred to Plan 3.5+ (later if photographer uses it):**
- Native single-file binary (Rust/Go) replacing the .bat/.command pair — no terminal flash on click.
- Tray/menubar icon for clean Quit.
- First-launch GUI installer dialog (currently: Jack runs install once on her laptop).
- Codesigning to bypass SmartScreen / Gatekeeper warnings.

---

## File Structure

**Create:**
- `scripts/launcher/install_runtime.py` — one-shot CLI that copies a built runtime to its canonical local path. Idempotent.
- `scripts/launcher/stage_folder.py` — drops `Pixsage Search.bat` + `Pixsage Search.command` into a target folder.
- `scripts/launcher/launcher_templates.py` — string constants holding the .bat and .command bodies. Pure data; testable.
- `tests/launcher/test_launcher_templates.py`
- `tests/launcher/test_stage_folder.py`
- `tests/launcher/test_install_runtime.py`

**Modify:**
- `src/pixsage/cli.py` — add `pixsage stage-launchers <photo_root>` verb that invokes `stage_folder.py`'s logic.

---

### Task 1: Launcher templates (string constants)

**Files:**
- Create: `scripts/launcher/launcher_templates.py`
- Create: `tests/launcher/test_launcher_templates.py`

- [ ] **Step 1: Write the failing test**

Create `tests/launcher/test_launcher_templates.py`:

```python
from scripts.launcher.launcher_templates import WINDOWS_BAT, MACOS_COMMAND, render


def test_windows_bat_invokes_runtime_pythonw_and_serves_parent_dir() -> None:
    body = render(WINDOWS_BAT, runtime_path=r"%LOCALAPPDATA%\pixsage")
    assert "%LOCALAPPDATA%\\pixsage\\python\\pythonw.exe" in body
    assert "-m pixsage serve" in body
    assert "%~dp0" in body  # parent dir of the .bat


def test_macos_command_invokes_runtime_python_and_serves_parent_dir() -> None:
    body = render(MACOS_COMMAND, runtime_path="$HOME/Library/Application Support/pixsage")
    assert "$HOME/Library/Application Support/pixsage/python/bin/python3" in body
    assert "-m pixsage serve" in body
    assert 'cd "$(dirname "$0")"' in body


def test_render_substitutes_only_known_placeholder() -> None:
    """render() does plain string substitution for {runtime_path}, nothing else."""
    template = "echo {runtime_path} and {other}"
    out = render(template, runtime_path="X")
    assert out == "echo X and {other}"
```

- [ ] **Step 2: Run failing test**

Run: `pytest tests/launcher/test_launcher_templates.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement launcher_templates.py**

Create `scripts/launcher/launcher_templates.py`:

```python
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
start "" "{runtime_path}\python\pythonw.exe" -m pixsage serve "%~dp0"
"""


MACOS_COMMAND = r"""#!/bin/bash
# Pixsage Search launcher (macOS).
# Runs the locally-installed pixsage runtime against this folder.
cd "$(dirname "$0")"
exec "{runtime_path}/python/bin/python3" -m pixsage serve "$PWD"
"""


def render(template: str, runtime_path: str) -> str:
    """Substitute {runtime_path} into a template. No other placeholders."""
    return template.replace("{runtime_path}", runtime_path)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/launcher/test_launcher_templates.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run full suite**

Run: `pytest -x --tb=no -q`
Expected: 205 passed (202 + 3 new), 2 skipped, 1 xfailed.

- [ ] **Step 6: Commit**

```bash
git add scripts/launcher/launcher_templates.py tests/launcher/test_launcher_templates.py
git commit -m "feat(launcher): launcher_templates — Windows .bat + macOS .command bodies

Pure-data string templates with one placeholder ({runtime_path}).
Both invoke pythonw/python -m pixsage serve <containing-folder>.
serve already opens the browser; no extra launcher logic needed."
```

---

### Task 2: `stage_folder.py` — drop launchers into a folder

**Files:**
- Create: `scripts/launcher/stage_folder.py`
- Create: `tests/launcher/test_stage_folder.py`

- [ ] **Step 1: Write the failing test**

Create `tests/launcher/test_stage_folder.py`:

```python
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
    # Owner exec bit set (mode & 0o100). On Windows this stat may not
    # reflect POSIX exec semantics, but the chmod call should still succeed
    # without raising. Tolerate that mode comparison can be a no-op on Win.
    import sys
    if sys.platform != "win32":
        assert mode & stat.S_IXUSR, f"missing +x on {cmd}"


def test_stage_folder_idempotent(tmp_path: Path) -> None:
    from scripts.launcher.stage_folder import stage_folder

    folder = tmp_path / "shoot"
    folder.mkdir()
    stage_folder(folder)
    stage_folder(folder)  # second call should not raise
    assert (folder / "Pixsage Search.bat").exists()
    assert (folder / "Pixsage Search.command").exists()


def test_stage_folder_rejects_missing_dir(tmp_path: Path) -> None:
    from scripts.launcher.stage_folder import stage_folder

    nope = tmp_path / "does-not-exist"
    with pytest.raises((FileNotFoundError, NotADirectoryError)):
        stage_folder(nope)
```

- [ ] **Step 2: Run failing test**

Run: `pytest tests/launcher/test_stage_folder.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement stage_folder.py**

Create `scripts/launcher/stage_folder.py`:

```python
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

    # +x for macOS launch-on-double-click. On Windows this is a no-op for
    # POSIX bits but doesn't fail.
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/launcher/test_stage_folder.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run full suite**

Run: `pytest -x --tb=no -q`
Expected: 209 passed (205 + 4 new), 2 skipped, 1 xfailed.

- [ ] **Step 6: Commit**

```bash
git add scripts/launcher/stage_folder.py tests/launcher/test_stage_folder.py
git commit -m "feat(launcher): stage_folder.py — drop launchers into an indexed folder

Writes both .bat + .command from templates, sets +x on the .command
for macOS launch-on-double-click. Idempotent."
```

---

### Task 3: `pixsage stage-launchers` CLI verb

**Files:**
- Modify: `src/pixsage/cli.py`
- Create: `tests/test_cli_stage_launchers.py`

**Background:** Convenience verb so Jack can run `pixsage stage-launchers E:\Sony alpha 7c` instead of remembering the script path. Imports `stage_folder` from the build pipeline.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_stage_launchers.py`:

```python
from pathlib import Path

from typer.testing import CliRunner

from pixsage.cli import app


def test_stage_launchers_writes_files(tmp_path: Path) -> None:
    folder = tmp_path / "Sony alpha 7c"
    folder.mkdir()
    runner = CliRunner()
    result = runner.invoke(app, ["stage-launchers", str(folder)])
    assert result.exit_code == 0, result.stdout
    assert (folder / "Pixsage Search.bat").exists()
    assert (folder / "Pixsage Search.command").exists()


def test_stage_launchers_rejects_missing_dir(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["stage-launchers", str(tmp_path / "missing")])
    # typer.Argument(... exists=True) makes this exit 2 (usage error)
    assert result.exit_code != 0
```

- [ ] **Step 2: Run failing test**

Run: `pytest tests/test_cli_stage_launchers.py -v`
Expected: FAIL — `stage-launchers` is not a known command.

- [ ] **Step 3: Add the verb to cli.py**

Read `src/pixsage/cli.py` to find a good insertion point — somewhere after the existing verbs like `embed`, `geolocate`, `serve`. Append a new `@app.command()` block:

```python
@app.command(name="stage-launchers")
def stage_launchers(
    photo_root: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, resolve_path=True),
) -> None:
    """Drop `Pixsage Search.bat` + `Pixsage Search.command` into an indexed folder.

    Run once per folder after `pixsage embed` so the photographer can launch
    the search webapp by double-clicking the launcher in Explorer / Finder.

    Requires the pixsage runtime to already be installed at the canonical
    local path (%LOCALAPPDATA%\\pixsage on Windows, ~/Library/Application
    Support/pixsage on Mac). See `scripts/launcher/install_runtime.py`.
    """
    from scripts.launcher.stage_folder import stage_folder
    stage_folder(photo_root)
    typer.echo(f"Staged launchers in: {photo_root}")
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_cli_stage_launchers.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run full suite**

Run: `pytest -x --tb=no -q`
Expected: 211 passed (209 + 2 new), 2 skipped, 1 xfailed.

- [ ] **Step 6: Commit**

```bash
git add src/pixsage/cli.py tests/test_cli_stage_launchers.py
git commit -m "feat(cli): pixsage stage-launchers verb

Convenience wrapper around scripts.launcher.stage_folder.stage_folder
so Jack can run \`pixsage stage-launchers E:\\path\` to drop the
clickable .bat + .command into an indexed folder."
```

---

### Task 4: `install_runtime.py` — one-shot runtime install on the photographer's machine

**Files:**
- Create: `scripts/launcher/install_runtime.py`
- Create: `tests/launcher/test_install_runtime.py`

**Background:** On the photographer's laptop, run this once to put the runtime at the canonical local path. Two modes:
1. `--from-tarball <path>` — extract a pre-built tarball (Plan 2's runtime + models, packed into one .tar.gz)
2. `--build-now` — invoke build_runtime + download_models live (slow, requires internet, but no manual tarball)

For v1 we ship just `--build-now` since it's what we have. The tarball mode is a Plan 3.5 follow-up.

- [ ] **Step 1: Write the failing test**

Create `tests/launcher/test_install_runtime.py`:

```python
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
```

- [ ] **Step 2: Run failing test**

Run: `pytest tests/launcher/test_install_runtime.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement install_runtime.py**

Create `scripts/launcher/install_runtime.py`:

```python
"""One-shot runtime install for the photographer's machine.

Run once (e.g. by Jack on her laptop, or by her from a setup script):

    python -m scripts.launcher.install_runtime --target windows-x64

Puts a portable Python + pixsage[serve] + pre-staged HF models under
the canonical platform path:
- Windows: %LOCALAPPDATA%\\pixsage\\
- macOS:   ~/Library/Application Support/pixsage/

After this, `pixsage stage-launchers <folder>` drops a clickable
`Pixsage Search` into each indexed folder, which the runtime serves.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from scripts.launcher.build_runtime import build_runtime
from scripts.launcher.download_models import download_models


def canonical_install_path(home_override: Path | None = None) -> Path:
    """Return the OS-specific canonical install dir for the pixsage runtime."""
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        if not local:
            raise RuntimeError("LOCALAPPDATA env var is not set")
        return Path(local) / "pixsage"
    # macOS / linux: use ~/Library/Application Support on darwin; on linux
    # default to ~/.local/share/pixsage to follow XDG. We only support darwin
    # for v1; linux is unsupported but we don't reject — same path style works.
    home = home_override or Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "pixsage"
    return home / ".local" / "share" / "pixsage"


def install_runtime_via_build(
    install_dir: Path,
    target: str,
    force: bool = False,
) -> None:
    """Invoke build_runtime + download_models into `install_dir`."""
    if (install_dir / "python").exists() and not force:
        print(f"Runtime already at {install_dir}; skipping (use --force to reinstall)")
        return
    print(f"Installing runtime to: {install_dir}")
    build_runtime(target_name=target, out_dir=install_dir)
    download_models(out_dir=install_dir)
    print(f"\nRuntime ready at: {install_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Install pixsage runtime locally.")
    parser.add_argument("--target", required=True, choices=["windows-x64", "macos-arm64"])
    parser.add_argument("--install-dir", type=Path, default=None,
                        help="Override install location (default: canonical OS path).")
    parser.add_argument("--force", action="store_true", help="Reinstall even if a runtime exists.")
    args = parser.parse_args()

    install_dir = args.install_dir or canonical_install_path()
    install_runtime_via_build(install_dir=install_dir, target=args.target, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/launcher/test_install_runtime.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run full suite**

Run: `pytest -x --tb=no -q`
Expected: 214 passed (211 + 3 new), 2 skipped, 1 xfailed.

- [ ] **Step 6: Commit**

```bash
git add scripts/launcher/install_runtime.py tests/launcher/test_install_runtime.py
git commit -m "feat(launcher): install_runtime.py — one-shot local runtime install

Run once on the photographer's machine to put a portable Python +
pixsage[serve] + pre-staged HF models at the canonical platform path
(%LOCALAPPDATA%\\pixsage on Windows, ~/Library/Application Support/pixsage
on macOS). Idempotent — skips if already installed unless --force."
```

---

### Task 5: Verification + journal

- [ ] **Step 1: Confirm test counts**

Run: `pytest --tb=no -q`
Expected: 214 passed, 2 skipped, 1 xfailed.

- [ ] **Step 2: Confirm CLIs respond**

Run: `python -m scripts.launcher.stage_folder --help`
Run: `python -m scripts.launcher.install_runtime --help`
Run: `pixsage stage-launchers --help`
All three should print argparse/typer usage.

- [ ] **Step 3: Document the photographer-onboarding flow**

Append to README.md (or create `docs/launcher-setup.md`) a 5-line setup checklist:
1. On the photographer's machine: `python -m scripts.launcher.install_runtime --target windows-x64` (one time, ~10 minutes).
2. On Jack's workstation: `pixsage stage-launchers E:\Sony alpha 7c` (one per indexed folder).
3. Photographer plugs in drive, opens folder, double-clicks `Pixsage Search.bat`.
4. Browser opens to localhost; she searches.
5. To stop: close the browser tab + kill the python process via Task Manager (no tray icon yet).

- [ ] **Step 4: Optional manual smoke**

Stage a folder under `tests/demo_corpus` and click the `.bat` to verify end-to-end. (Skip if you've already validated the parts independently.)

---

## Self-review

**Spec coverage:**
- §"Per-folder native launcher" — replaced with .bat/.command shell wrappers (pragmatic v1; native binary deferred).
- §"Local runtime path constant per OS" ✅ (Task 4).
- §"First-time-setup flow" — replaced with a one-shot CLI Jack runs on her machine. No GUI dialog (Plan 3.5).
- §"Browser open after server up" — handled by existing `pixsage serve` (cli.py:609-611). No change needed.
- §"Tray icon, codesigning, atomic install with sha256" — explicitly deferred.

**Placeholder scan:** none.

**Type consistency:** `Path` everywhere. `target: str` matches Plan 2's convention.

**Risk:** Terminal-window flash on Windows is tolerated by using `pythonw.exe` (windowless Python) plus `start ""`. If python-build-standalone's tarball doesn't include `pythonw.exe`, the .bat falls back to a brief cmd window which is acceptable.
