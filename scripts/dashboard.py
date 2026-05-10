"""Live progress dashboard for the pixsage pipeline.

Run alongside a `pixsage tag/embed/geolocate/export` run to monitor progress,
system load, and per-stage throughput. Polls every 2 seconds.

Required: `psutil` (`pip install psutil`); `nvidia-smi` on PATH for GPU stats.

The orchestrating shell script is expected to redirect each stage's stdout/
stderr to `<logdir>/<stage>.log` so the dashboard can read the live tqdm
progress fragment from the most recently updated log. See the Monitor command
inside the project's full-run kickoff for an example.

    python scripts/dashboard.py "E:/Sony alpha 7c" \\
        --logdir C:/path/to/full-run-logs \\
        --total-raw-paths 2123 \\
        --port 8766
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import pyarrow.parquet as pq
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse


PHOTOINDEX: Path
LOGDIR: Path
TOTAL_RAW_PATHS: int = 0
DUPE_RATE: float = 0.0  # fraction of paths estimated to be byte-duplicates of others
START_TS = time.time()


def _gpu_stats() -> dict | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode != 0:
            return None
        parts = [p.strip() for p in result.stdout.strip().split("\n")[0].split(",")]
        return {
            "name": parts[0],
            "util_percent": float(parts[1]),
            "mem_used_mb": float(parts[2]),
            "mem_total_mb": float(parts[3]),
            "temperature_c": float(parts[4]),
            "power_watts": float(parts[5]) if parts[5] and parts[5] != "[N/A]" else None,
        }
    except Exception:
        return None


def _catalog_stats() -> dict:
    cat_path = PHOTOINDEX / "catalog.db"
    if not cat_path.exists():
        return {"photos": 0, "tagged": 0, "captioned": 0, "geolocated": 0, "errored": 0}
    con = sqlite3.connect(f"file:{cat_path}?mode=ro", uri=True)
    try:
        photos = con.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
        tagged = con.execute(
            "SELECT COUNT(*) FROM photos WHERE last_tagged_at IS NOT NULL"
        ).fetchone()[0]
        captioned = con.execute(
            "SELECT COUNT(*) FROM photos WHERE caption IS NOT NULL"
        ).fetchone()[0]
        try:
            geolocated = con.execute(
                "SELECT COUNT(DISTINCT sha256) FROM geo_predictions"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            geolocated = 0
        errored = con.execute(
            "SELECT COUNT(*) FROM photos WHERE error_reason IS NOT NULL"
        ).fetchone()[0]
        return {
            "photos": photos, "tagged": tagged, "captioned": captioned,
            "geolocated": geolocated, "errored": errored,
        }
    finally:
        con.close()


def _vector_stats() -> dict:
    img_path = PHOTOINDEX / "vectors" / "siglip2_image.parquet"
    cap_path = PHOTOINDEX / "vectors" / "minilm_caption.parquet"
    img_n = pq.ParquetFile(img_path).metadata.num_rows if img_path.exists() else 0
    cap_n = pq.ParquetFile(cap_path).metadata.num_rows if cap_path.exists() else 0
    return {"image_vecs": img_n, "caption_vecs": cap_n}


def _active_stage() -> str:
    if not LOGDIR.exists():
        return "idle"
    logs = list(LOGDIR.glob("*.log"))
    if not logs:
        return "idle"
    most_recent = max(logs, key=lambda p: p.stat().st_mtime)
    age = time.time() - most_recent.stat().st_mtime
    if age > 60:
        return "idle"  # nothing recently updated
    return most_recent.stem


def _last_progress_line(stage: str) -> str:
    log_path = LOGDIR / f"{stage}.log"
    if not log_path.exists():
        return ""
    try:
        size = log_path.stat().st_size
        with open(log_path, "rb") as f:
            f.seek(max(0, size - 8192))
            tail = f.read().decode("utf-8", errors="replace")
        last = tail.replace("\r", "\n").strip().split("\n")[-1]
        return last[:240]
    except Exception:
        return ""


_disk_anchor: dict | None = None


def _disk_read_mbs() -> float:
    global _disk_anchor
    counters = psutil.disk_io_counters()
    if counters is None:
        return 0.0
    now = time.time()
    if _disk_anchor is None:
        _disk_anchor = {"ts": now, "read_bytes": counters.read_bytes}
        return 0.0  # need a second sample to compute a rate
    dt = max(0.001, now - _disk_anchor["ts"])
    delta = max(0, counters.read_bytes - _disk_anchor["read_bytes"])
    _disk_anchor = {"ts": now, "read_bytes": counters.read_bytes}
    return delta / dt / 1024 / 1024


_history: dict[str, list[tuple[float, int]]] = {"tag": [], "embed": [], "geolocate": []}


def _throughput(stage: str, current: int) -> tuple[float, int]:
    """photos/sec over a 60s rolling window for the given stage's counter, plus ETA seconds."""
    now = time.time()
    h = _history.setdefault(stage, [])
    h.append((now, current))
    _history[stage] = [(t, c) for t, c in h if now - t < 60]
    h = _history[stage]
    if len(h) < 2:
        return 0.0, 0
    t0, c0 = h[0]
    dt = max(0.001, now - t0)
    rate = max(0, (current - c0) / dt)
    return rate, 0  # ETA computed in caller


app = FastAPI(title="pixsage dashboard")


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>pixsage pipeline</title>
<style>
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; margin: 0; padding: 1.5em; }
h1 { color: #58a6ff; margin: 0 0 0.2em 0; font-size: 1.6em; }
.subtitle { color: #8b949e; font-size: 0.85em; margin-bottom: 1.5em; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5em; max-width: 1200px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1.2em 1.4em; }
.card h2 { margin: 0 0 0.8em 0; font-size: 0.95em; color: #58a6ff; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.row { display: flex; justify-content: space-between; align-items: center; padding: 0.35em 0; border-bottom: 1px solid #21262d; gap: 1em; }
.row:last-child { border-bottom: none; }
.label { color: #8b949e; font-size: 0.9em; }
.value { font-family: 'SF Mono', 'Cascadia Code', Consolas, monospace; font-size: 0.9em; color: #e6edf3; }
.bar { height: 16px; background: #21262d; border-radius: 3px; overflow: hidden; margin: 0.4em 0 0.6em 0; position: relative; }
.bar-fill { height: 100%; background: linear-gradient(90deg, #1f6feb, #58a6ff); transition: width 0.6s ease; }
.bar-fill.gpu { background: linear-gradient(90deg, #2ea043, #56d364); }
.bar-fill.cpu { background: linear-gradient(90deg, #d29922, #e3b341); }
.bar-text { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; font-size: 0.7em; color: #f0f6fc; text-shadow: 0 0 4px rgba(0,0,0,0.7); font-family: monospace; }
.stage-pill { padding: 0.3em 0.8em; border-radius: 12px; font-size: 0.8em; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.stage-pill.tag { background: #1f6feb; color: white; }
.stage-pill.embed { background: #2ea043; color: white; }
.stage-pill.geolocate { background: #a371f7; color: white; }
.stage-pill.export { background: #db6d28; color: white; }
.stage-pill.idle { background: #30363d; color: #8b949e; }
.tail { font-family: 'SF Mono', Consolas, monospace; font-size: 0.7em; color: #6e7681; padding: 0.5em 0.8em; word-break: break-all; max-height: 4.5em; overflow: hidden; background: #0d1117; border-radius: 4px; margin-top: 0.4em; }
.metric-big { font-size: 1.4em; font-family: monospace; color: #f0f6fc; }
@media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>pixsage pipeline</h1>
<div class="subtitle" id="header">connecting…</div>
<div id="root"></div>
<script>
async function refresh() {
  try {
    const r = await fetch('/api/state');
    const s = await r.json();
    document.getElementById('header').textContent =
      `${s.photo_root} · dashboard up ${s.elapsed}`;
    document.getElementById('root').innerHTML = render(s);
  } catch (e) {
    document.getElementById('header').textContent = 'disconnected — retrying…';
  }
}
function bar(pct, label, cls) {
  const p = Math.min(100, Math.max(0, pct));
  return `<div class="bar"><div class="bar-fill ${cls||''}" style="width:${p.toFixed(1)}%"></div><div class="bar-text">${label}</div></div>`;
}
function fmtETA(secs) {
  if (!secs || !isFinite(secs) || secs <= 0) return '—';
  const h = Math.floor(secs/3600), m = Math.floor((secs%3600)/60), s = Math.floor(secs%60);
  return h > 0 ? `${h}h ${m}m` : (m > 0 ? `${m}m ${s}s` : `${s}s`);
}
function escapeHtml(s) { return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function render(s) {
  const tagDenom = s.expected_unique || 1;
  const tagPct = Math.min(100, s.tagged / tagDenom * 100);
  const embedPct = s.tagged > 0 ? Math.min(100, s.image_vecs / s.tagged * 100) : 0;
  const geoPct = s.tagged > 0 ? Math.min(100, s.geolocated / s.tagged * 100) : 0;
  const stage = s.stage;
  const stageClass = ['tag','embed','geolocate','export'].includes(stage) ? stage : 'idle';
  const gpuMemPct = s.gpu ? (s.gpu.mem_used_mb / s.gpu.mem_total_mb * 100) : 0;
  return `
    <div class="grid">
      <div class="card">
        <h2>active stage</h2>
        <div class="row"><span class="label">stage</span><span class="stage-pill ${stageClass}">${stage}</span></div>
        <div class="row"><span class="label">throughput</span><span class="value">${s.throughput_per_sec.toFixed(2)} photos/s</span></div>
        <div class="row"><span class="label">stage ETA</span><span class="value">${fmtETA(s.eta_seconds)}</span></div>
        <div class="row"><span class="label">model runs / dupe writes / errored</span><span class="value">${s.tagged} / ${s.dupe_writes_estimate} / ${s.errored}</span></div>
        <div class="row"><span class="label">last log fragment</span></div>
        <div class="tail">${escapeHtml(s.last_line)}</div>
      </div>
      <div class="card">
        <h2>pipeline progress</h2>
        <div class="row"><span class="label">tag</span><span class="value">${s.tagged} / ~${s.expected_unique} unique shas</span></div>
        ${bar(tagPct, tagPct.toFixed(1) + '%')}
        <div class="row"><span class="label">embed (image)</span><span class="value">${s.image_vecs} / ${s.tagged}</span></div>
        ${bar(embedPct, embedPct.toFixed(1) + '%')}
        <div class="row"><span class="label">embed (caption)</span><span class="value">${s.caption_vecs} / ${s.captioned}</span></div>
        ${bar(s.captioned > 0 ? Math.min(100, s.caption_vecs/s.captioned*100) : 0, '')}
        <div class="row"><span class="label">geolocate</span><span class="value">${s.geolocated} / ${s.tagged}</span></div>
        ${bar(geoPct, geoPct.toFixed(1) + '%')}
      </div>
      <div class="card">
        <h2>system</h2>
        <div class="row"><span class="label">cpu</span><span class="value">${s.cpu_percent.toFixed(1)}%</span></div>
        ${bar(s.cpu_percent, '', 'cpu')}
        <div class="row"><span class="label">ram</span><span class="value">${s.ram_used_gb.toFixed(1)} / ${s.ram_total_gb.toFixed(1)} GB</span></div>
        ${bar(s.ram_percent, '')}
        ${s.gpu ? `
          <div class="row"><span class="label">${s.gpu.name}</span><span class="value">${s.gpu.util_percent.toFixed(0)}% · ${s.gpu.temperature_c.toFixed(0)}°C${s.gpu.power_watts ? ' · ' + s.gpu.power_watts.toFixed(0) + 'W' : ''}</span></div>
          ${bar(s.gpu.util_percent, '', 'gpu')}
          <div class="row"><span class="label">gpu memory</span><span class="value">${(s.gpu.mem_used_mb/1024).toFixed(1)} / ${(s.gpu.mem_total_mb/1024).toFixed(1)} GB</span></div>
          ${bar(gpuMemPct, '', 'gpu')}
        ` : '<div class="row"><span class="label">gpu</span><span class="value">n/a</span></div>'}
        <div class="row"><span class="label">disk read (system-wide)</span><span class="value">${s.disk_read_mbs.toFixed(1)} MB/s</span></div>
      </div>
      <div class="card">
        <h2>catalog</h2>
        <div class="row"><span class="label">photos in catalog</span><span class="value">${s.photos}</span></div>
        <div class="row"><span class="label">tagged</span><span class="value">${s.tagged}</span></div>
        <div class="row"><span class="label">captioned</span><span class="value">${s.captioned}</span></div>
        <div class="row"><span class="label">image vectors</span><span class="value">${s.image_vecs}</span></div>
        <div class="row"><span class="label">caption vectors</span><span class="value">${s.caption_vecs}</span></div>
        <div class="row"><span class="label">geo predictions</span><span class="value">${s.geolocated} photos × top-5</span></div>
        <div class="row"><span class="label">errored</span><span class="value" style="color: ${s.errored > 0 ? '#f85149' : '#e6edf3'}">${s.errored}</span></div>
      </div>
    </div>
  `;
}
refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.get("/api/state")
def state() -> JSONResponse:
    cat = _catalog_stats()
    vec = _vector_stats()
    gpu = _gpu_stats()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    vm = psutil.virtual_memory()
    stage = _active_stage()
    last_line = _last_progress_line(stage) if stage != "idle" else ""

    elapsed_s = time.time() - START_TS
    h = int(elapsed_s // 3600); m = int((elapsed_s % 3600) // 60); sec = int(elapsed_s % 60)
    elapsed_str = f"{h:02d}:{m:02d}:{sec:02d}"

    # Expected unique shas: start from the (1 - dupe-rate) fraction of total
    # raw paths, then refine to the actual catalog photos count once the
    # hash+upsert pass has populated most of the catalog (catalog row count is
    # the authoritative unique-sha count).
    unique_estimate = int(TOTAL_RAW_PATHS * (1.0 - DUPE_RATE))
    if cat["photos"] > 0.95 * unique_estimate:
        expected_unique = cat["photos"]
    else:
        expected_unique = max(cat["photos"], unique_estimate)

    # Throughput: which counter is moving depends on stage.
    if stage == "tag":
        rate, _ = _throughput("tag", cat["tagged"])
        remaining = max(0, expected_unique - cat["tagged"])
    elif stage == "embed":
        rate, _ = _throughput("embed", vec["image_vecs"])
        remaining = max(0, cat["tagged"] - vec["image_vecs"])
    elif stage == "geolocate":
        rate, _ = _throughput("geolocate", cat["geolocated"])
        remaining = max(0, cat["tagged"] - cat["geolocated"])
    else:
        rate, remaining = 0.0, 0
    eta_seconds = (remaining / rate) if rate > 0 else 0

    dupe_writes_estimate = max(0, TOTAL_RAW_PATHS - cat["photos"]) if stage == "tag" else 0

    return JSONResponse({
        "photo_root": str(PHOTOINDEX.parent),
        "stage": stage,
        "elapsed": elapsed_str,
        "last_line": last_line,
        **cat,
        **vec,
        "expected_unique": expected_unique,
        "dupe_writes_estimate": dupe_writes_estimate,
        "throughput_per_sec": rate,
        "eta_seconds": eta_seconds,
        "cpu_percent": cpu_percent,
        "ram_used_gb": (vm.total - vm.available) / 1024**3,
        "ram_total_gb": vm.total / 1024**3,
        "ram_percent": vm.percent,
        "gpu": gpu,
        "disk_read_mbs": _disk_read_mbs(),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("photo_root", type=Path)
    parser.add_argument("--logdir", type=Path, required=True,
                        help="Directory containing tag.log / embed.log / geolocate.log / export.log")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--total-raw-paths", type=int, default=0,
                        help="Total raw paths in the corpus. Required for ETA "
                             "during the tag stage (catalog photo count alone "
                             "doesn't tell us how many shas are still pending).")
    parser.add_argument("--dupe-rate", type=float, default=0.0,
                        help="Fraction of paths estimated to be byte-duplicates "
                             "of others (0.0–1.0). Used for the early-stage "
                             "unique-sha estimate before the catalog stabilizes. "
                             "Defaults to 0 (assume no duplicates).")
    args = parser.parse_args()

    global PHOTOINDEX, LOGDIR, TOTAL_RAW_PATHS, DUPE_RATE
    PHOTOINDEX = args.photo_root / ".photoindex"
    LOGDIR = args.logdir
    TOTAL_RAW_PATHS = args.total_raw_paths
    DUPE_RATE = max(0.0, min(0.99, args.dupe_rate))

    print(f"dashboard at http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
