"""Build a portable Python runtime tree at <out>/.

This script handles the three steps of producing a usable `<out>/python/`
directory:
  1. Download the python-build-standalone tarball for the target (cached).
  2. Extract it into <out>/.
  3. Verify the resulting python binary runs and reports the expected version.

A separate step (pip install pixsage[serve] --target) is added in Task 4 of
the runtime-build plan.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

from scripts.launcher.pbs_targets import get_target


def download_pbs_tarball(url: str, cache_dir: Path) -> Path:
    """Download the PBS tarball if not already in cache. Returns local path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(url).path).name
    dest = cache_dir / filename
    if dest.exists():
        return dest
    print(f"Downloading {url} -> {dest}")
    urlretrieve(url, str(dest))
    return dest


def extract_pbs_tarball(tarball: Path, runtime_dir: Path) -> None:
    """Extract `tarball` into `runtime_dir`. Creates runtime_dir if needed."""
    runtime_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {tarball.name} -> {runtime_dir}")
    with tarfile.open(tarball, "r:*") as tar:
        # Use filter='data' (Python 3.12+) to safely strip absolute paths /
        # device files / dangerous metadata.
        try:
            tar.extractall(runtime_dir, filter="data")
        except TypeError:
            tar.extractall(runtime_dir)  # 3.11 fallback


def verify_python_binary(python_exe: Path, expected_version: str) -> None:
    """Run `python_exe -V` and assert it matches `expected_version` (e.g. "3.12.6")."""
    if not python_exe.exists():
        raise RuntimeError(f"python binary missing at {python_exe}")
    result = subprocess.run(
        [str(python_exe), "-V"],
        capture_output=True,
        text=True,
        check=True,
    )
    # `python -V` writes "Python X.Y.Z" to stdout (3.4+) or stderr (older).
    output = (result.stdout + result.stderr).strip()
    if expected_version not in output:
        raise RuntimeError(
            f"version mismatch: {python_exe} reports {output!r}, expected {expected_version!r}"
        )
    print(f"  python OK: {output}")


def pip_install(
    python_exe: Path,
    target_dir: Path,
    project_dir: Path,
    extras: str = "serve",
) -> None:
    """Install the pixsage project + extras into target_dir using the runtime's python.

    `target_dir` becomes the on-disk equivalent of a site-packages — the runtime
    consumes it via PYTHONPATH=<target_dir> at launch time.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    project_spec = f"{project_dir}[{extras}]"
    cmd = [
        str(python_exe),
        "-m", "pip", "install",
        "--target", str(target_dir),
        "--upgrade",
        project_spec,
    ]
    print(f"  pip install -> {target_dir}")
    subprocess.run(cmd, check=True)


def build_runtime(
    target_name: str,
    out_dir: Path,
    cache_dir: Path | None = None,
    project_dir: Path | None = None,
    skip_pip: bool = False,
) -> Path:
    """Run the full download->extract->verify->pip-install pipeline. Returns the python binary path."""
    target = get_target(target_name)
    cache_dir = cache_dir or (Path.home() / ".cache" / "pixsage-launcher-build")
    project_dir = project_dir or Path(__file__).resolve().parents[2]

    tarball = download_pbs_tarball(target.tarball_url, cache_dir)
    extract_pbs_tarball(tarball, out_dir)

    python_exe = out_dir / target.python_relpath
    verify_python_binary(python_exe, target.expected_python_version)

    if not skip_pip:
        pip_install(
            python_exe=python_exe,
            target_dir=out_dir / "site-packages",
            project_dir=project_dir,
            extras="serve",
        )
    return python_exe


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a portable pixsage runtime.")
    parser.add_argument("--target", required=True, choices=sorted(["windows-x64", "macos-arm64"]))
    parser.add_argument("--out", required=True, type=Path, help="Output directory.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Tarball cache.")
    parser.add_argument("--skip-pip", action="store_true", help="Skip the pip install step (download+extract only).")
    args = parser.parse_args()

    python_exe = build_runtime(args.target, args.out, args.cache_dir, skip_pip=args.skip_pip)
    print(f"\nRuntime built at: {args.out}")
    print(f"Python binary:     {python_exe}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
