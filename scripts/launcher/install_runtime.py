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
