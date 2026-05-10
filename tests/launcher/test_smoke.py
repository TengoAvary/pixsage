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

    # 4. Seed the catalog with dummy rows so serve has something to render.
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
        # 6. Poll for readiness (~90s budget; SigLIP2 cold load is ~8s on CPU,
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
