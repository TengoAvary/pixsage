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
import subprocess
import sys
from pathlib import Path

from scripts.launcher.build_runtime import build_runtime
from scripts.launcher.pbs_targets import TARGETS, get_target


def canonical_install_path(home_override: Path | None = None) -> Path:
    """Return the OS-specific canonical install dir for the pixsage runtime."""
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        if not local:
            raise RuntimeError("LOCALAPPDATA env var is not set")
        return Path(local) / "pixsage"
    home = home_override or Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "pixsage"
    return home / ".local" / "share" / "pixsage"


def install_runtime_via_build(
    install_dir: Path,
    target: str,
    force: bool = False,
    project_dir: Path | None = None,
) -> None:
    """Invoke build_runtime + download_models into `install_dir`.

    Model downloads run via the freshly-extracted runtime python so the
    bootstrap python this script runs on needs only the stdlib (no
    huggingface_hub install required).
    """
    if (install_dir / "python").exists() and not force:
        print(f"Runtime already at {install_dir}; skipping (use --force to reinstall)")
        return
    project_dir = project_dir or Path(__file__).resolve().parents[2]
    print(f"Installing runtime to: {install_dir}")
    build_runtime(target_name=target, out_dir=install_dir, project_dir=project_dir)

    python_exe = install_dir / get_target(target).python_relpath
    env = os.environ.copy()
    # Runtime site-packages holds huggingface_hub (via [serve] → transformers);
    # project_dir on PYTHONPATH lets the subprocess find scripts.launcher.*.
    env["PYTHONPATH"] = os.pathsep.join([
        str(install_dir / "site-packages"),
        str(project_dir),
    ])
    env["PYTHONNOUSERSITE"] = "1"
    subprocess.run(
        [str(python_exe), "-m", "scripts.launcher.download_models",
         "--out", str(install_dir)],
        env=env,
        check=True,
    )
    _install_laptop_launcher(install_dir)
    print(f"\nRuntime ready at: {install_dir}")


def _install_laptop_launcher(install_dir: Path) -> None:
    """Drop a single laptop-level Pixsage Search launcher.

    Mac: ~/Applications/Pixsage Search.command
    Win: %USERPROFILE%\\Desktop\\Pixsage Search.bat
    """
    from scripts.launcher.launcher_templates import (
        LAPTOP_MACOS_COMMAND,
        LAPTOP_WINDOWS_BAT,
        render,
    )
    if sys.platform == "darwin":
        target_dir = Path.home() / "Applications"
        target_dir.mkdir(exist_ok=True)
        target = target_dir / "Pixsage Search.command"
        target.write_text(render(LAPTOP_MACOS_COMMAND, runtime_path=str(install_dir)))
        target.chmod(0o755)
    elif sys.platform == "win32":
        target = Path.home() / "Desktop" / "Pixsage Search.bat"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render(LAPTOP_WINDOWS_BAT, runtime_path=str(install_dir)))
    else:
        return
    print(f"Laptop launcher: {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Install pixsage runtime locally.")
    parser.add_argument("--target", required=True, choices=sorted(TARGETS.keys()))
    parser.add_argument("--install-dir", type=Path, default=None,
                        help="Override install location (default: canonical OS path).")
    parser.add_argument("--force", action="store_true", help="Reinstall even if a runtime exists.")
    args = parser.parse_args()

    install_dir = args.install_dir or canonical_install_path()
    install_runtime_via_build(install_dir=install_dir, target=args.target, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
