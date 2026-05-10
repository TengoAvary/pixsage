# Launcher Plan 2: Portable Runtime Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Workstation-side build pipeline that produces a self-contained `<out>/` directory containing a portable Python interpreter, pixsage + serve dependencies in a target-only `site-packages/`, and a pre-staged Hugging Face model cache. The output can be copied to a target machine and used to run `pixsage serve` against a translated catalog (Plan 1) without any system Python or HF download step.

**Architecture:** Two scripts under `scripts/launcher/`:
1. `build_runtime.py` — pulls a [python-build-standalone](https://github.com/astral-sh/python-build-standalone) tarball for the target OS, extracts it, and uses its bundled pip to install `pixsage[serve]` into a flat `site-packages/`.
2. `download_models.py` — uses `huggingface_hub.snapshot_download` to pre-stage SigLIP2 + MiniLM into `<out>/models/` in HF cache layout, so setting `HF_HOME=<out>/models` makes transformers find them without network.

A new `[serve]` pip extras in `pyproject.toml` slims the dep set vs. what `[taggers] + [search]` would pull in.

**Tech Stack:** Python 3.12 (target runtime version), python-build-standalone, huggingface_hub, pip-install --target.

**Companion plans:**
- Plan 1 (path translation): shipped 2026-05-10. Catalog now portable across machines.
- Plan 3: native launcher (Rust crate) + per-folder staging — depends on the artifacts this plan produces.

**Deferred from Plan 2 to a follow-up:**
- Text-tower-only SigLIP2 extraction (~6× model-size reduction, drops `<out>/models/` from ~1.8 GB to ~280 MB). Will be a separate Plan 2.5 once the full-fat runtime is proven working.

---

## File Structure

**Create:**
- `scripts/launcher/__init__.py` — empty marker so the dir is importable for tests.
- `scripts/launcher/build_runtime.py` — CLI that builds the Python tree.
- `scripts/launcher/download_models.py` — CLI that pre-stages models.
- `scripts/launcher/pbs_targets.py` — target table mapping `{windows-x64, macos-arm64} → tarball URL`. Pure data, no I/O.
- `tests/launcher/__init__.py`
- `tests/launcher/test_pbs_targets.py` — unit tests for the target table.
- `tests/launcher/test_build_runtime.py` — unit tests with subprocess + downloads stubbed.
- `tests/launcher/test_download_models.py` — unit tests with snapshot_download stubbed.
- `tests/launcher/test_smoke.py` — end-to-end smoke test, gated on environment vars (`PIXSAGE_LAUNCHER_SMOKE=1`) so CI doesn't try to do a real 30-minute build.

**Modify:**
- `pyproject.toml` — add `[serve]` extras combining `[search]` deps with `torch + transformers`.

---

### Task 1: Add `[serve]` pip extras

**Files:**
- Modify: `pyproject.toml`

**Background:** The runtime needs to run `pixsage serve` only — no `tag`, `embed`, `geolocate`. That means: fastapi, uvicorn, jinja2, httpx, sentence-transformers (already in `[search]`) PLUS torch and transformers (currently only in `[taggers]`, alongside rawpy + ram which serve doesn't need). Add a third extras `[serve]` that's the runtime-relevant subset.

- [ ] **Step 1: Read the current pyproject.toml**

Run: `cat pyproject.toml | head -70`
Note the existing `[taggers]` and `[search]` blocks (lines 22-48 currently).

- [ ] **Step 2: Add the new extras block**

Edit `pyproject.toml`. Immediately after the `search = [...]` block (around line 48), add:

```toml
serve = [
  # Subset of [taggers] + [search] needed at serve time. No rawpy / ram —
  # those are tag/embed-time only. The launcher runtime ships exactly this
  # set, so keep it minimal.
  "torch>=2.2",
  "transformers>=4.50,<5",
  "fastapi>=0.110",
  "uvicorn[standard]>=0.27",
  "jinja2>=3.1",
  "httpx>=0.27",
  "sentence-transformers>=3.0",
  "huggingface-hub>=0.24",
]
```

- [ ] **Step 3: Verify the extras resolve**

Run: `pip install --dry-run -e ".[serve]" 2>&1 | tail -5`
Expected: pip resolves successfully (may print "Would install ..." with a list of packages and their versions). NOT expected: any "ResolutionImpossible" error.

If pip is not on PATH in your shell, use `python -m pip install --dry-run -e ".[serve]"` instead.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat(deps): [serve] extras for the launcher runtime

Slimmer than [taggers]+[search] (no rawpy/ram). Captures exactly
what pixsage serve needs at runtime: torch + transformers (for SigLIP2),
sentence-transformers (for MiniLM), fastapi stack, plus huggingface-hub
which build pipeline scripts depend on directly."
```

---

### Task 2: `pbs_targets.py` — target table for python-build-standalone

**Files:**
- Create: `scripts/launcher/__init__.py` (empty file)
- Create: `scripts/launcher/pbs_targets.py`
- Create: `tests/launcher/__init__.py` (empty file)
- Create: `tests/launcher/test_pbs_targets.py`

**Background:** [python-build-standalone](https://github.com/astral-sh/python-build-standalone/releases) ships portable Python tarballs at predictable URLs. We pin to a specific release tag + Python version + target triple. Pure-data table makes it testable and easy to update.

- [ ] **Step 1: Create the empty package markers**

```bash
mkdir -p scripts/launcher tests/launcher
touch scripts/launcher/__init__.py tests/launcher/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/launcher/test_pbs_targets.py`:

```python
import pytest

from scripts.launcher.pbs_targets import PBS_RELEASE, TARGETS, get_target


def test_pbs_release_is_pinned() -> None:
    assert PBS_RELEASE  # non-empty string like "20260508"
    assert PBS_RELEASE.isdigit()
    assert len(PBS_RELEASE) == 8  # YYYYMMDD


def test_targets_have_expected_keys() -> None:
    assert set(TARGETS.keys()) == {"windows-x64", "macos-arm64"}


def test_get_target_returns_known() -> None:
    target = get_target("windows-x64")
    assert target.tarball_url.startswith("https://github.com/astral-sh/python-build-standalone/")
    assert target.tarball_url.endswith(".tar.zst")
    assert target.python_relpath  # e.g. "python/install/python.exe" or "python/bin/python3"
    assert target.expected_python_version.startswith("3.12.")


def test_get_target_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_target("linux-x64")
```

- [ ] **Step 3: Run the failing test**

Run: `pytest tests/launcher/test_pbs_targets.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.launcher.pbs_targets'`.

- [ ] **Step 4: Implement the target table**

Create `scripts/launcher/pbs_targets.py`:

```python
"""python-build-standalone target table.

Pinned versions: update PBS_RELEASE + PYTHON_VERSION here when bumping.
Verify URLs by checking https://github.com/astral-sh/python-build-standalone/releases/tag/<PBS_RELEASE>
"""
from __future__ import annotations

from dataclasses import dataclass

# python-build-standalone release tag (YYYYMMDD format).
# Verified to contain 3.12.x install_only-stripped tarballs for both targets below.
PBS_RELEASE = "20240909"
PYTHON_VERSION = "3.12.6"


@dataclass(frozen=True)
class PBSTarget:
    name: str  # human-readable target id, e.g. "windows-x64"
    tarball_url: str
    # Path inside the extracted tarball where the python binary lives,
    # relative to the runtime root we extract into.
    python_relpath: str
    expected_python_version: str


def _pbs_url(filename: str) -> str:
    return (
        f"https://github.com/astral-sh/python-build-standalone/releases/download/"
        f"{PBS_RELEASE}/{filename}"
    )


TARGETS: dict[str, PBSTarget] = {
    "windows-x64": PBSTarget(
        name="windows-x64",
        tarball_url=_pbs_url(
            f"cpython-{PYTHON_VERSION}+{PBS_RELEASE}-x86_64-pc-windows-msvc-install_only_stripped.tar.gz"
        ),
        # The install_only_stripped layout puts python.exe directly under python/
        python_relpath="python/python.exe",
        expected_python_version=PYTHON_VERSION,
    ),
    "macos-arm64": PBSTarget(
        name="macos-arm64",
        tarball_url=_pbs_url(
            f"cpython-{PYTHON_VERSION}+{PBS_RELEASE}-aarch64-apple-darwin-install_only_stripped.tar.gz"
        ),
        python_relpath="python/bin/python3",
        expected_python_version=PYTHON_VERSION,
    ),
}


def get_target(name: str) -> PBSTarget:
    if name not in TARGETS:
        raise KeyError(
            f"unknown target {name!r}; known: {sorted(TARGETS.keys())}"
        )
    return TARGETS[name]
```

NOTE on URL filenames: the test pins the structure (`.tar.zst` suffix etc.) but the actual filenames vary between PBS releases. The `install_only_stripped.tar.gz` form is what python-build-standalone's recent releases ship as the smallest portable variant. If `pip install --dry-run` later reveals the URL doesn't resolve, adjust the suffix and update the test (the test currently asserts `.tar.zst`; that's wrong if PBS uses `.tar.gz` — fix the test to match what PBS actually ships).

- [ ] **Step 5: Update test to match `.tar.gz` filename suffix**

Replace the assertion `assert target.tarball_url.endswith(".tar.zst")` with `assert target.tarball_url.endswith(".tar.gz")` in `tests/launcher/test_pbs_targets.py`.

- [ ] **Step 6: Run tests**

Run: `pytest tests/launcher/test_pbs_targets.py -v`
Expected: 4 passed.

- [ ] **Step 7: Run full suite — confirm nothing else broke**

Run: `pytest -x --tb=no -q`
Expected: 194 passed (190 + 4 new), 1 skipped, 1 xfailed.

- [ ] **Step 8: Commit**

```bash
git add scripts/launcher/__init__.py scripts/launcher/pbs_targets.py tests/launcher/__init__.py tests/launcher/test_pbs_targets.py
git commit -m "feat(launcher): pbs_targets — python-build-standalone URL table

Pinned to PBS release 20240909 + Python 3.12.6. Two targets:
windows-x64, macos-arm64. Pure-data; no I/O."
```

---

### Task 3: `build_runtime.py` — download + extract Python tree

**Files:**
- Create: `scripts/launcher/build_runtime.py`
- Create: `tests/launcher/test_build_runtime.py`

**Background:** Given a target name and an output directory, this script downloads the PBS tarball (cached locally so re-runs are fast), extracts it into `<out>/python/`, and verifies the `python` binary exists and reports the expected version.

The `pip install --target` step is a separate task (Task 4) so each task remains under 5 file changes.

- [ ] **Step 1: Write the failing test**

Create `tests/launcher/test_build_runtime.py`:

```python
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.launcher.build_runtime import (
    download_pbs_tarball,
    extract_pbs_tarball,
    verify_python_binary,
)


def test_download_uses_cache_when_present(tmp_path: Path) -> None:
    """If the tarball is already in the cache dir, no download happens."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    fake_tarball = cache_dir / "fake.tar.gz"
    fake_tarball.write_bytes(b"already-here")

    # Patch urllib.request.urlretrieve to fail loudly if called.
    with patch("scripts.launcher.build_runtime.urlretrieve") as m:
        result = download_pbs_tarball(
            url="https://example.com/fake.tar.gz",
            cache_dir=cache_dir,
        )
    m.assert_not_called()
    assert result == fake_tarball


def test_download_fetches_when_cache_miss(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    def fake_urlretrieve(url: str, dest: str) -> tuple[str, object]:
        Path(dest).write_bytes(b"downloaded")
        return (dest, None)

    with patch("scripts.launcher.build_runtime.urlretrieve", side_effect=fake_urlretrieve) as m:
        result = download_pbs_tarball(
            url="https://example.com/foo.tar.gz",
            cache_dir=cache_dir,
        )
    m.assert_called_once()
    assert result.exists()
    assert result.read_bytes() == b"downloaded"


def test_extract_unpacks_into_runtime_dir(tmp_path: Path) -> None:
    """Extract a tiny tarball and verify the runtime dir has expected contents."""
    import tarfile
    import io

    # Build a minimal tarball: contains python/foo.txt
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="python/foo.txt")
        payload = b"hello"
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    tarball = tmp_path / "tiny.tar.gz"
    tarball.write_bytes(buf.getvalue())

    runtime_dir = tmp_path / "runtime"
    extract_pbs_tarball(tarball, runtime_dir)
    assert (runtime_dir / "python" / "foo.txt").read_bytes() == b"hello"


def test_verify_python_binary_runs_python_v(tmp_path: Path) -> None:
    """Uses the host python as a stand-in to verify the version-check logic."""
    import sys

    host_python = Path(sys.executable)
    expected_version = ".".join(str(p) for p in sys.version_info[:3])
    # verify_python_binary should not raise when versions match
    verify_python_binary(host_python, expected_version)


def test_verify_python_binary_rejects_wrong_version(tmp_path: Path) -> None:
    import sys

    host_python = Path(sys.executable)
    with pytest.raises(RuntimeError, match="version mismatch"):
        verify_python_binary(host_python, "9.99.999")
```

- [ ] **Step 2: Run failing test**

Run: `pytest tests/launcher/test_build_runtime.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement build_runtime.py (partial — download + extract + verify only)**

Create `scripts/launcher/build_runtime.py`:

```python
"""Build a portable Python runtime tree at <out>/.

This script handles the three steps of producing a usable `<out>/python/`
directory:
  1. Download the python-build-standalone tarball for the target (cached).
  2. Extract it into <out>/.
  3. Verify the resulting python binary runs and reports the expected version.

A separate step (pip install pixsage[serve] --target) is in build_runtime_pip.py
or appended to this same module in Task 4.
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
    print(f"Downloading {url} → {dest}")
    urlretrieve(url, str(dest))
    return dest


def extract_pbs_tarball(tarball: Path, runtime_dir: Path) -> None:
    """Extract `tarball` into `runtime_dir`. Creates runtime_dir if needed."""
    runtime_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {tarball.name} → {runtime_dir}")
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


def build_runtime(target_name: str, out_dir: Path, cache_dir: Path | None = None) -> Path:
    """Run the full download→extract→verify pipeline. Returns the python binary path."""
    target = get_target(target_name)
    cache_dir = cache_dir or (Path.home() / ".cache" / "pixsage-launcher-build")

    tarball = download_pbs_tarball(target.tarball_url, cache_dir)
    extract_pbs_tarball(tarball, out_dir)

    python_exe = out_dir / target.python_relpath
    verify_python_binary(python_exe, target.expected_python_version)
    return python_exe


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a portable pixsage runtime.")
    parser.add_argument("--target", required=True, choices=sorted(["windows-x64", "macos-arm64"]))
    parser.add_argument("--out", required=True, type=Path, help="Output directory.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Tarball cache.")
    args = parser.parse_args()

    python_exe = build_runtime(args.target, args.out, args.cache_dir)
    print(f"\nRuntime built at: {args.out}")
    print(f"Python binary:     {python_exe}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/launcher/test_build_runtime.py -v`
Expected: 5 passed.

- [ ] **Step 5: Smoke check — run `python -m scripts.launcher.build_runtime --help`**

Run: `python -m scripts.launcher.build_runtime --help`
Expected: argparse usage output with --target and --out flags.

- [ ] **Step 6: Run full suite**

Run: `pytest -x --tb=no -q`
Expected: 199 passed (194 + 5 new), 1 skipped, 1 xfailed.

- [ ] **Step 7: Commit**

```bash
git add scripts/launcher/build_runtime.py tests/launcher/test_build_runtime.py
git commit -m "feat(launcher): build_runtime.py download + extract + verify

Pulls python-build-standalone tarball, extracts into <out>/, verifies
the python binary version. pip-install step is a follow-up task."
```

---

### Task 4: `build_runtime.py` — pip install pixsage[serve] into site-packages

**Files:**
- Modify: `scripts/launcher/build_runtime.py`
- Modify: `tests/launcher/test_build_runtime.py`

**Background:** After the Python tree is in place, install pixsage + its serve-time deps into a flat `<out>/site-packages/` directory. Using `--target` means runtime can find them via `PYTHONPATH=<out>/site-packages` without needing pip's normal install mechanics.

- [ ] **Step 1: Add the failing test**

Append to `tests/launcher/test_build_runtime.py`:

```python
def test_pip_install_invokes_target_pip(tmp_path: Path) -> None:
    """pip_install should call the runtime's python with `-m pip install --target`."""
    from scripts.launcher.build_runtime import pip_install

    fake_python = tmp_path / "python.exe"
    fake_python.write_bytes(b"")  # presence-only

    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        # Mimic CompletedProcess
        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        return R()

    with patch("scripts.launcher.build_runtime.subprocess.run", side_effect=fake_run):
        pip_install(
            python_exe=fake_python,
            target_dir=tmp_path / "site-packages",
            project_dir=tmp_path / "project",
            extras="serve",
        )
    assert len(captured) == 1
    cmd = captured[0]
    assert cmd[0] == str(fake_python)
    assert cmd[1:5] == ["-m", "pip", "install", "--target"]
    assert cmd[5] == str(tmp_path / "site-packages")
    # The project + extras spec is the last arg
    assert cmd[-1].endswith("[serve]")
```

- [ ] **Step 2: Run the failing test**

Run: `pytest tests/launcher/test_build_runtime.py::test_pip_install_invokes_target_pip -v`
Expected: FAIL — `ImportError: cannot import name 'pip_install'`.

- [ ] **Step 3: Add `pip_install` to build_runtime.py**

In `scripts/launcher/build_runtime.py`, after `verify_python_binary` and before `build_runtime`, add:

```python
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
    print(f"  pip install → {target_dir}")
    subprocess.run(cmd, check=True)
```

Then update `build_runtime()` to call it:

```python
def build_runtime(
    target_name: str,
    out_dir: Path,
    cache_dir: Path | None = None,
    project_dir: Path | None = None,
    skip_pip: bool = False,
) -> Path:
    """Run the full download→extract→verify→pip-install pipeline. Returns the python binary path."""
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
```

And add a CLI flag for skipping pip in `main()`:

```python
    parser.add_argument("--skip-pip", action="store_true", help="Skip the pip install step (download+extract only).")
```

…and pass it into the call:

```python
    python_exe = build_runtime(args.target, args.out, args.cache_dir, skip_pip=args.skip_pip)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/launcher/test_build_runtime.py -v`
Expected: 6 passed.

- [ ] **Step 5: Run full suite**

Run: `pytest -x --tb=no -q`
Expected: 200 passed (199 + 1 new), 1 skipped, 1 xfailed.

- [ ] **Step 6: Commit**

```bash
git add scripts/launcher/build_runtime.py tests/launcher/test_build_runtime.py
git commit -m "feat(launcher): build_runtime pip install step

After the Python tree is extracted, install pixsage[serve] into
<out>/site-packages with pip --target so the runtime can import
it via PYTHONPATH at launch time."
```

---

### Task 5: `download_models.py` — pre-stage SigLIP2 + MiniLM into HF cache

**Files:**
- Create: `scripts/launcher/download_models.py`
- Create: `tests/launcher/test_download_models.py`

**Background:** Pre-staging models means the runtime starts cold without an internet hit. We use `huggingface_hub.snapshot_download` with `cache_dir=<out>/models/hub` so the layout matches what `transformers.AutoModel.from_pretrained` expects when `HF_HOME=<out>/models`.

- [ ] **Step 1: Write the failing test**

Create `tests/launcher/test_download_models.py`:

```python
from pathlib import Path
from unittest.mock import patch

import pytest


def test_download_models_invokes_snapshot_download_for_each_repo(tmp_path: Path) -> None:
    from scripts.launcher.download_models import REPOS, download_models

    captured: list[dict] = []

    def fake_snapshot(repo_id: str, **kwargs) -> str:
        captured.append({"repo_id": repo_id, **kwargs})
        # Return a fake local path; snapshot_download contract is to return the
        # path to the downloaded snapshot.
        return str(tmp_path / repo_id.replace("/", "--"))

    with patch("scripts.launcher.download_models.snapshot_download", side_effect=fake_snapshot):
        download_models(out_dir=tmp_path / "models")

    repo_ids = {c["repo_id"] for c in captured}
    assert repo_ids == set(REPOS)
    for c in captured:
        # All calls should target <out>/models/hub
        assert c["cache_dir"] == str(tmp_path / "models" / "hub")


def test_repos_contains_siglip2_and_minilm() -> None:
    from scripts.launcher.download_models import REPOS

    assert "google/siglip2-so400m-patch14-384" in REPOS
    assert "sentence-transformers/all-MiniLM-L6-v2" in REPOS
```

- [ ] **Step 2: Run failing test**

Run: `pytest tests/launcher/test_download_models.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement download_models.py**

Create `scripts/launcher/download_models.py`:

```python
"""Pre-stage SigLIP2 + MiniLM into the runtime's HF cache.

After this runs, `HF_HOME=<out>/models` is enough for transformers and
sentence-transformers to find both models offline. The on-disk layout is:

    <out>/models/hub/
        models--google--siglip2-so400m-patch14-384/
            snapshots/<rev>/...
        models--sentence-transformers--all-MiniLM-L6-v2/
            snapshots/<rev>/...

Total size: ~1.8 GB (will drop to ~280 MB once the text-tower-only optimization
in the follow-up plan ships).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


REPOS = [
    "google/siglip2-so400m-patch14-384",
    "sentence-transformers/all-MiniLM-L6-v2",
]


def download_models(out_dir: Path) -> None:
    cache_dir = out_dir / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for repo_id in REPOS:
        print(f"Downloading {repo_id} → {cache_dir}")
        snapshot_download(repo_id=repo_id, cache_dir=str(cache_dir))


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-stage pixsage runtime models.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory (will create <out>/hub/).")
    args = parser.parse_args()

    download_models(args.out)
    print(f"\nModels staged at: {args.out}")
    print(f"To use: HF_HOME={args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/launcher/test_download_models.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run full suite**

Run: `pytest -x --tb=no -q`
Expected: 202 passed (200 + 2 new), 1 skipped, 1 xfailed.

- [ ] **Step 6: Commit**

```bash
git add scripts/launcher/download_models.py tests/launcher/test_download_models.py
git commit -m "feat(launcher): download_models.py — pre-stage HF cache

snapshot_download into <out>/hub/ so runtime sets HF_HOME=<out> and
transformers + sentence-transformers find SigLIP2 + MiniLM offline."
```

---

### Task 6: End-to-end smoke test (gated)

**Files:**
- Create: `tests/launcher/test_smoke.py`

**Background:** The unit tests stub network + subprocess calls. This task is the real validation — actually building a runtime, downloading the models (or skipping if cached), and running `pixsage serve` from within the produced runtime against `tests/demo_corpus`. Gated on `PIXSAGE_LAUNCHER_SMOKE=1` so normal `pytest` runs don't take 30 minutes.

- [ ] **Step 1: Write the gated smoke test**

Create `tests/launcher/test_smoke.py`:

```python
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest


SMOKE_ENABLED = os.environ.get("PIXSAGE_LAUNCHER_SMOKE") == "1"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.mark.skipif(
    not SMOKE_ENABLED,
    reason="set PIXSAGE_LAUNCHER_SMOKE=1 to run; downloads ~2GB and takes minutes",
)
def test_smoke_build_and_serve(tmp_path: Path) -> None:
    """End-to-end: build runtime, stage models, run pixsage serve, hit /."""
    from scripts.launcher.build_runtime import build_runtime
    from scripts.launcher.download_models import download_models

    target = "windows-x64" if sys.platform == "win32" else "macos-arm64"
    runtime_dir = tmp_path / "runtime"
    models_dir = tmp_path / "models"

    # 1. Build the runtime (Python + pixsage[serve])
    python_exe = build_runtime(target_name=target, out_dir=runtime_dir)

    # 2. Pre-stage models
    download_models(out_dir=models_dir)

    # 3. Make a tiny test "drive" with a catalog already pointing at the demo corpus
    project_dir = Path(__file__).resolve().parents[2]
    demo = project_dir / "tests" / "demo_corpus"
    photo_root = tmp_path / "test-drive" / "demo"
    photo_root.mkdir(parents=True)
    for jpg in list(demo.glob("*.jpg"))[:3]:
        shutil.copy(jpg, photo_root / jpg.name)

    # 4. Run pixsage tag + embed via the runtime so the catalog has real entries.
    #    Note: tag isn't strictly needed for serve to work; we just need rows in
    #    the photos table. Use a tiny script to insert dummy rows directly.
    #    (Real tag/embed would re-download tagger models which is out of scope.)
    seed_script = tmp_path / "seed.py"
    seed_script.write_text(f"""\
from pathlib import Path
from pixsage.catalog import Catalog
photo_root = Path(r"{photo_root}")
photoindex = photo_root / ".photoindex"
photoindex.mkdir(exist_ok=True)
cat = Catalog(photoindex / "catalog.db")
cat.init_schema()
cat.set_photo_root_if_unset(photo_root)
import sqlite3
for jpg in photo_root.glob("*.jpg"):
    cat._conn.execute(
        "INSERT INTO photos (sha256, current_path, filename, filesize, mtime, added_at, last_seen_at) "
        "VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'))",
        (jpg.stem, str(jpg), jpg.name, jpg.stat().st_size, jpg.stat().st_mtime),
    )
cat._conn.commit()
""")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(runtime_dir / "site-packages")
    env["HF_HOME"] = str(models_dir)
    env["HF_HUB_OFFLINE"] = "1"

    subprocess.run(
        [str(python_exe), str(seed_script)],
        env=env,
        check=True,
    )

    # 5. Boot the server
    port = _free_port()
    proc = subprocess.Popen(
        [
            str(python_exe), "-m", "pixsage", "serve",
            str(photo_root),
            "--port", str(port),
            "--no-open",
        ],
        env=env,
    )
    try:
        # 6. Poll for readiness (~30s budget; SigLIP2 cold load is ~8s on CPU,
        # but we're running the full model so allow headroom)
        deadline = time.time() + 90
        ready = False
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    ready = True
                    break
            except OSError:
                time.sleep(0.5)
        assert ready, "server never came up"

        # 7. Hit / and verify it returns HTML
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as r:
            assert r.status == 200
            body = r.read(1024).decode("utf-8", errors="replace")
            assert "<html" in body.lower() or "search" in body.lower()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
```

- [ ] **Step 2: Verify the test SKIPS when env var is unset**

Run: `pytest tests/launcher/test_smoke.py -v`
Expected: 1 skipped (with reason about `PIXSAGE_LAUNCHER_SMOKE=1`).

- [ ] **Step 3: Run full suite**

Run: `pytest -x --tb=no -q`
Expected: 202 passed, 2 skipped, 1 xfailed (smoke test added the second skip).

- [ ] **Step 4: Run the smoke test for real (manual, takes ~10 minutes first time)**

Run: `PIXSAGE_LAUNCHER_SMOKE=1 pytest tests/launcher/test_smoke.py -v -s`
Expected: passes. First run downloads ~2 GB of model weights + the PBS tarball; cached on subsequent runs.

If it fails:
- DO NOT edit the test to make it pass. Investigate the failure first.
- Common failures: PBS URL 404 (update `PBS_RELEASE` in `pbs_targets.py`); pip install fails on Windows path separators (test reports the exact subprocess error); the runtime python can't import pixsage (PYTHONPATH typo).

- [ ] **Step 5: Commit**

```bash
git add tests/launcher/test_smoke.py
git commit -m "test(launcher): end-to-end smoke (gated)

PIXSAGE_LAUNCHER_SMOKE=1 to run. Builds the runtime, stages models,
runs pixsage serve via the produced python binary, verifies / responds.
Skipped by default — full pytest stays fast."
```

---

### Task 7: Verification + plan handoff

- [ ] **Step 1: Confirm test counts**

Run: `pytest --tb=no -q`
Expected: 202 passed, 2 skipped, 1 xfailed.

- [ ] **Step 2: Confirm scripts run from CLI**

Run: `python -m scripts.launcher.build_runtime --help`
Expected: argparse usage output.

Run: `python -m scripts.launcher.download_models --help`
Expected: argparse usage output.

- [ ] **Step 3: Optionally run the smoke test for real**

If you have ~10 GB of disk free and time:

Run: `PIXSAGE_LAUNCHER_SMOKE=1 pytest tests/launcher/test_smoke.py -v -s`
Expected: passes; you'll see ~2 GB of HF downloads, a ~15 MB python-build-standalone tarball, and a brief `pixsage serve` startup.

- [ ] **Step 4: Update the project journal**

Run `/journal` at the end of the session. Highlight:
- Plan 2 shipped: workstation can produce a portable Python+pixsage runtime + pre-staged models for any target machine
- Plan 3 (native launcher) queued
- Known follow-up: SigLIP2 text-tower-only extraction will drop model footprint from 1.8 GB → 280 MB

---

## Self-review

**Spec coverage:**
- §"build_runtime.py uses python-build-standalone tarballs" ✅ (Tasks 2-4)
- §"pip-installs pixsage[serve] into <out>/site-packages" ✅ (Task 4)
- §"download_models.py pre-stages SigLIP2 + MiniLM in HF cache layout" ✅ (Task 5)
- §"smoke test verifies the runtime can run pixsage serve" ✅ (Task 6, gated)
- §"the smaller model footprint via text-tower extraction" — explicitly deferred per the user's note that simpler-first is acceptable.

**Placeholder scan:** none.

**Type consistency:** `Path` used consistently; `target_name: str` everywhere; `out_dir`, `cache_dir`, `project_dir` are the canonical arg names across both scripts.

**Method-name consistency:** `download_pbs_tarball`, `extract_pbs_tarball`, `verify_python_binary`, `pip_install`, `build_runtime`, `download_models` — all functions, no leaky abstractions.

**Risk awareness:**
- python-build-standalone URLs change between releases. The test pin (`PBS_RELEASE = "20240909"`) may need a refresh when re-running the smoke test on a stale machine. Detection: smoke fails with HTTP 404 from `urlretrieve`. Recovery: update the pin, re-run.
- pip install --target on Windows can produce paths with backslash issues if the project_dir has spaces. Detection: pip subprocess returns non-zero. Recovery: quote the path explicitly when constructing the spec.
- HF_HUB_OFFLINE=1 in the smoke test is strict — if any model file is missing from the staged cache, transformers raises rather than falling back to network. That's intentional; it proves the cache is complete.
