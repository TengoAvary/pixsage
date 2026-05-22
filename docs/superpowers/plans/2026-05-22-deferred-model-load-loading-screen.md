# Deferred Model Load + Loading Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `pixsage serve` accept requests in <1s by moving the ~12s model+vector load off the startup path into a background thread, with a full-page loading screen that shows phased progress and auto-loads the search UI when ready.

**Architecture:** A new `BackendLoader` owns readiness state (`loading`/`ready`/`error` + phases). `build_app()` splits into a fast synchronous half (registry, app, routes, empty state containers) and a slow `load_fn` (embedder + per-catalog services) run either inline (`defer_load=False`, default — preserves today's behavior for tests) or in a daemon thread (`defer_load=True`, used by `cli.serve`). A `/status` endpoint + a readiness middleware + a loading-branch in `/` gate the app until ready.

**Tech Stack:** FastAPI/Starlette, Jinja2, threading (stdlib), pytest + `fastapi.testclient.TestClient`.

---

### Task 1: `BackendLoader` state machine

**Files:**
- Create: `src/pixsage/web/loader.py`
- Test: `tests/test_loader.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_loader.py`:

```python
from __future__ import annotations

import pytest

from pixsage.web.loader import BackendLoader


def test_initial_state_is_loading_with_pending_phases():
    loader = BackendLoader(["Loading search model…", "Loading catalog vectors…"])
    snap = loader.snapshot()
    assert snap["status"] == "loading"
    assert snap["error"] is None
    assert [p["label"] for p in snap["phases"]] == [
        "Loading search model…",
        "Loading catalog vectors…",
    ]
    assert all(p["state"] == "pending" for p in snap["phases"])


def test_run_advances_phases_and_becomes_ready():
    loader = BackendLoader(["a", "b"])
    seen = []

    def load_fn(ldr):
        ldr.start_phase(0)
        seen.append(ldr.snapshot()["phases"][0]["state"])  # active
        ldr.finish_phase(0)
        ldr.start_phase(1)
        ldr.finish_phase(1)

    loader.run(load_fn)
    snap = loader.snapshot()
    assert seen == ["active"]
    assert snap["status"] == "ready"
    assert [p["state"] for p in snap["phases"]] == ["done", "done"]


def test_run_records_error_and_leaves_active_phase_visible():
    loader = BackendLoader(["a", "b"])

    def load_fn(ldr):
        ldr.start_phase(0)
        raise RuntimeError("boom")

    loader.run(load_fn)
    snap = loader.snapshot()
    assert snap["status"] == "error"
    assert "boom" in snap["error"]
    assert "RuntimeError" in snap["error"]
    # The phase that was active when it failed is still reported as active.
    assert snap["phases"][0]["state"] == "active"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pixsage.web.loader'`

- [ ] **Step 3: Write the implementation**

Create `src/pixsage/web/loader.py`:

```python
from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class _Phase:
    label: str
    state: str = "pending"  # pending | active | done


class BackendLoader:
    """Tracks readiness of the search backend while it warms up.

    status: "loading" -> "ready", or "loading" -> "error".
    Thread-safe: the background load thread mutates phase/status under a lock;
    request handlers read a consistent snapshot via ``snapshot()``.
    """

    def __init__(self, phase_labels: list[str]) -> None:
        self._phases = [_Phase(label) for label in phase_labels]
        self.status = "loading"
        self.error: str | None = None
        self._lock = threading.Lock()

    def start_phase(self, index: int) -> None:
        with self._lock:
            self._phases[index].state = "active"

    def finish_phase(self, index: int) -> None:
        with self._lock:
            self._phases[index].state = "done"

    def run(self, load_fn) -> None:
        """Execute load_fn(self); flip to ready on success, error on exception.

        On failure the phase that was active is left as-is (so the loading
        screen shows which step failed) and status/error are set."""
        try:
            load_fn(self)
        except Exception as e:  # noqa: BLE001 — surface any load failure to UI
            with self._lock:
                self.error = f"{type(e).__name__}: {e}"
                self.status = "error"
            return
        with self._lock:
            self.status = "ready"

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "error": self.error,
                "phases": [{"label": p.label, "state": p.state} for p in self._phases],
            }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_loader.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/web/loader.py tests/test_loader.py
git commit -m "feat(web): BackendLoader readiness state machine"
```

---

### Task 2: Split `build_app` into sync half + deferred `load_fn`

**Files:**
- Modify: `src/pixsage/web/app.py` (replace the body of `build_app`, lines ~50–165)
- Modify: `src/pixsage/cli.py` (the `serve` command, the `build_app(...)` call)
- Test: `tests/test_web_app.py` (add deferred-becomes-ready integration test)

Context: today `build_app` (app.py:50–165) does everything synchronously. The
per-catalog work is already factored into `routes._load_catalog_into_multi(app,
entry)` (routes.py:457), which reads `app.state.embedder`. We reuse it.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_web_app.py`:

```python
def test_deferred_load_eventually_becomes_ready(tmp_path: Path):
    """defer_load=True returns immediately in 'loading', then a background
    thread flips to 'ready' (mock embedder loads instantly)."""
    import time

    from pixsage.web.app import build_app

    app = build_app(
        registry_path=tmp_path / "catalogs.json",
        embedder_name="mock",
        defer_load=True,
    )
    # Returns before the load thread finishes -> starts as loading.
    assert app.state.loader.status in ("loading", "ready")

    deadline = time.time() + 5
    while time.time() < deadline and app.state.loader.status != "ready":
        time.sleep(0.05)
    assert app.state.loader.status == "ready"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_web_app.py::test_deferred_load_eventually_becomes_ready -v`
Expected: FAIL with `TypeError: build_app() got an unexpected keyword argument 'defer_load'`

- [ ] **Step 3: Rewrite `build_app` in `src/pixsage/web/app.py`**

Add `import threading` near the top imports (after `import tomllib`), add
`from pixsage.web.loader import BackendLoader` with the other `pixsage.*`
imports, then replace the entire `build_app` function body (lines 50–165) with:

```python
def build_app(
    photo_root: Path | None = None,
    registry_path: Path | None = None,
    embedder_name: str = "siglip2",
    *,
    defer_load: bool = False,
    experimental_cluster_labelling: bool = False,
) -> FastAPI:
    """Construct the FastAPI app for multi-catalog search.

    Args:
        photo_root: Optional. If given, ensures its .photoindex/ is in the
            registry (backward compat with the per-folder launcher model).
        registry_path: Override for the catalogs.json location.
        embedder_name: Which embedder to use for query encoding.
        defer_load: If True, load the embedder + catalog vectors in a background
            thread and return immediately (server answers a loading screen while
            it warms up). If False (default), load synchronously so the returned
            app is already ready — preserves behavior for tests and other callers.
        experimental_cluster_labelling: Off by default. See routes.py.
    """
    registry_path = registry_path or default_registry_path()
    registry = Registry(registry_path)
    registry.load()

    # Auto-register photo_root if given.
    if photo_root is not None:
        pi = photo_root / ".photoindex"
        pi.mkdir(parents=True, exist_ok=True)
        if registry.find_by_photoindex_path(str(pi.resolve())) is None:
            img_sig, cap_sig = derive_signatures(pi)
            registry.add(
                photoindex_path=str(pi.resolve()),
                label=photo_root.name,
                image_embedder_signature=img_sig,
                caption_embedder_signature=cap_sig,
            )

    # No startup discovery walk — catalogs enter the registry only via the
    # folder-browser picker (POST /catalogs/add-scan) or an explicit
    # photo_root arg. Startup only re-checks which registered paths exist.
    registry.refresh_availability()
    registry.save()

    # --- Synchronous half: cheap setup, returns near-instantly. ---
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app = FastAPI(title="pixsage")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    loader = BackendLoader(["Loading search model…", "Loading catalog vectors…"])
    app.state.loader = loader
    app.state.registry = registry
    app.state.registry_path = registry_path
    app.state.multi_search = MultiSearchService()
    app.state.embedder = None
    app.state.catalogs = {}                # {catalog_id: Catalog}
    app.state.path_resolvers = {}          # {catalog_id: PathResolver}
    app.state.thumbs_by_catalog = {}       # {catalog_id: ThumbnailCache}
    app.state.photoindex_paths = {}        # {catalog_id: Path}
    app.state.config = _default_config()   # replaced by load_fn once catalogs load
    app.state.templates = templates

    from pixsage.web import routes
    routes.register(app, experimental_cluster_labelling=experimental_cluster_labelling)

    # --- Slow half: embedder + per-catalog services. Run inline or threaded. ---
    def load_fn(ldr: BackendLoader) -> None:
        from pixsage.cli import _build_embedder
        from pixsage.device import select_device
        from pixsage.web.routes import _load_catalog_into_multi

        ldr.start_phase(0)
        embedder = _build_embedder(embedder_name)
        embedder.load(select_device())
        app.state.embedder = embedder
        ldr.finish_phase(0)

        ldr.start_phase(1)
        for entry in registry.entries():
            if not (entry.enabled and entry.available):
                continue
            _load_catalog_into_multi(app, entry)
        if app.state.catalogs:
            first_id = next(iter(app.state.catalogs))
            cfg_path = app.state.photoindex_paths[first_id] / "vocabulary.toml"
            ensure_default_config(cfg_path)
            app.state.config = load_config(cfg_path)
        ldr.finish_phase(1)

    if defer_load:
        threading.Thread(target=loader.run, args=(load_fn,), daemon=True).start()
    else:
        loader.run(load_fn)

    return app
```

- [ ] **Step 4: Update `cli.serve` to opt into deferral**

In `src/pixsage/cli.py`, change the `build_app(...)` call inside `serve`
(currently around line 800) to pass `defer_load=True`:

```python
    from pixsage.web.app import build_app
    fastapi_app = build_app(
        photo_root=photo_root,
        registry_path=registry,
        embedder_name=embedder,
        defer_load=True,
    )
```

(Leave the `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` `setdefault` lines above
this call untouched — they must still run before transformers imports.)

- [ ] **Step 5: Run the new test plus the existing web suite**

Run: `python3 -m pytest tests/test_web_app.py tests/test_web_app_multi.py tests/test_web_catalogs.py tests/test_web_search.py tests/test_web_thumbs.py tests/test_serve_path_translation.py -v`
Expected: PASS — the existing tests still pass (default `defer_load=False` keeps
synchronous behavior) and the new `test_deferred_load_eventually_becomes_ready`
passes.

- [ ] **Step 6: Commit**

```bash
git add src/pixsage/web/app.py src/pixsage/cli.py tests/test_web_app.py
git commit -m "feat(web): defer embedder+vector load off serve startup path"
```

---

### Task 3: `/status` endpoint, `/` loading branch, readiness middleware

**Files:**
- Modify: `src/pixsage/web/routes.py` (inside `register`, add imports, `/status`, `/` gate, middleware)
- Test: `tests/test_web_app.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_web_app.py`:

```python
def _ready_app(tmp_path: Path):
    from pixsage.web.app import build_app

    return build_app(
        registry_path=tmp_path / "catalogs.json",
        embedder_name="mock",
        defer_load=False,
    )


def test_status_endpoint_reports_ready(tmp_path: Path):
    app = _ready_app(tmp_path)
    with TestClient(app) as client:
        body = client.get("/status").json()
        assert body["status"] == "ready"
        assert all(p["state"] == "done" for p in body["phases"])
        assert body["error"] is None


def test_index_shows_loading_screen_when_not_ready(tmp_path: Path):
    app = _ready_app(tmp_path)
    app.state.loader.status = "loading"  # drive loading state deterministically
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "warming up" in r.text.lower() or "loading" in r.text.lower()
        assert client.get("/status").json()["status"] == "loading"


def test_gated_route_returns_503_while_loading_but_status_and_static_ok(tmp_path: Path):
    app = _ready_app(tmp_path)
    app.state.loader.status = "loading"
    with TestClient(app) as client:
        assert client.get("/thumb/cat/sha").status_code == 503
        assert client.get("/status").status_code == 200
        assert client.get("/static/htmx.min.js").status_code == 200


def test_index_shows_search_page_when_ready(tmp_path: Path):
    app = _ready_app(tmp_path)
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "search" in r.text.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_web_app.py -k "status_endpoint or loading_screen or 503 or search_page_when_ready" -v`
Expected: FAIL — `/status` 404, and `/` returns the search page (no loading
branch), `/thumb/...` returns 404 not 503.

- [ ] **Step 3: Add the route, gate, and middleware in `routes.py`**

In `src/pixsage/web/routes.py`, add `JSONResponse` to the existing fastapi
responses import (line 6):

```python
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
```

Inside `register(app, ...)`, immediately after the docstring and before the
existing `@app.get("/")` index route, add the middleware and status endpoint:

```python
    @app.middleware("http")
    async def _gate_until_ready(request, call_next):
        """Until the backend is ready, allow only the loading page, the status
        poll, and static assets; everything else 503s so no handler touches
        half-built app.state."""
        path = request.url.path
        if app.state.loader.status != "ready" and not (
            path == "/" or path == "/status" or path.startswith("/static")
        ):
            return JSONResponse(
                {"detail": "pixsage is still warming up"}, status_code=503
            )
        return await call_next(request)

    @app.get("/status")
    def status() -> JSONResponse:
        return JSONResponse(app.state.loader.snapshot())
```

Then add a readiness check as the very first lines inside the existing `index`
function body (right after `def index(... ) -> HTMLResponse:`):

```python
        if app.state.loader.status != "ready":
            return app.state.templates.TemplateResponse(
                request, "loading.html", {}
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_web_app.py -k "status_endpoint or loading_screen or 503 or search_page_when_ready" -v`
Expected: PASS (4 passed). Note `test_index_shows_loading_screen_when_not_ready`
and the 503 test require `loading.html` to exist — if it does not yet, they fail
on a template error. Create the template in Task 4 first if running these in
isolation; when executing in order, do Task 4 before re-running.

> Execution note: Tasks 3 and 4 are interdependent (the loading branch renders
> `loading.html`). Implement the route code (Task 3 Step 3) and the template
> (Task 4 Step 3) before running Task 3's full test set. Commit Task 3 after
> Task 4's template exists.

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/web/routes.py tests/test_web_app.py
git commit -m "feat(web): /status endpoint, loading-screen gate + readiness middleware"
```

---

### Task 4: Loading page template + styles

**Files:**
- Create: `src/pixsage/web/templates/loading.html`
- Modify: `src/pixsage/web/static/style.css` (append a loading block)
- Test: `tests/test_web_app.py` (template-content assertion)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_web_app.py`:

```python
def test_loading_page_has_poller_and_phase_markup(tmp_path: Path):
    app = _ready_app(tmp_path)
    app.state.loader.status = "loading"
    with TestClient(app) as client:
        html = client.get("/").text.lower()
        assert "/status" in html          # JS polls the status endpoint
        assert "pixsage" in html
        assert "phases" in html            # renders the phase checklist
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_web_app.py::test_loading_page_has_poller_and_phase_markup -v`
Expected: FAIL — `loading.html` does not exist (Jinja `TemplateNotFound`).

- [ ] **Step 3: Create `loading.html`**

Create `src/pixsage/web/templates/loading.html`:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>pixsage — starting…</title>
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body class="loading-page">
  <main class="loading-box">
    <h1>pixsage</h1>
    <div class="spinner" aria-hidden="true"></div>
    <p id="loading-headline">Warming up search models…</p>
    <ul id="phases"></ul>
    <p id="loading-error" hidden></p>
  </main>

  <script>
    const PHASE_GLYPH = { done: "●", active: "◑", pending: "○" };

    function render(data) {
      const ul = document.getElementById("phases");
      ul.innerHTML = "";
      for (const p of data.phases) {
        const li = document.createElement("li");
        li.className = "phase phase-" + p.state;
        li.textContent = (PHASE_GLYPH[p.state] || "○") + " " + p.label;
        ul.appendChild(li);
      }
    }

    async function poll() {
      let data;
      try {
        const r = await fetch("/status", { cache: "no-store" });
        data = await r.json();
      } catch (e) {
        // server not answering yet (race at very first ms) — retry shortly
        return setTimeout(poll, 500);
      }
      render(data);
      if (data.status === "ready") {
        location.reload();
        return;
      }
      if (data.status === "error") {
        document.getElementById("loading-headline").textContent =
          "Failed to start";
        const err = document.getElementById("loading-error");
        err.hidden = false;
        err.textContent = (data.error || "Unknown error") +
          " — check the terminal/logs.";
        return; // stop polling
      }
      setTimeout(poll, 500);
    }

    poll();
  </script>
</body>
</html>
```

- [ ] **Step 4: Append loading styles to `style.css`**

Append to `src/pixsage/web/static/style.css`:

```css
/* ── Loading screen ──────────────────────────────────────────────── */
.loading-page {
  display: flex;
  min-height: 100vh;
  margin: 0;
  align-items: center;
  justify-content: center;
  background: #f7f7f8;
  font-family: system-ui, -apple-system, sans-serif;
  color: #222;
}
.loading-box {
  text-align: center;
  max-width: 22rem;
}
.loading-box h1 { margin: 0 0 1.25rem; font-weight: 600; }
.spinner {
  width: 2.25rem;
  height: 2.25rem;
  margin: 0 auto 1rem;
  border: 3px solid #d8d8de;
  border-top-color: #555;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
#phases {
  list-style: none;
  padding: 0;
  margin: 1rem auto 0;
  display: inline-block;
  text-align: left;
}
.phase { padding: 0.15rem 0; color: #999; }
.phase-active { color: #222; font-weight: 600; }
.phase-done { color: #2a8a3e; }
#loading-error { color: #b00020; margin-top: 1rem; font-size: 0.9rem; }
```

- [ ] **Step 5: Run the loading-page test and the full Task-3 set**

Run: `python3 -m pytest tests/test_web_app.py -v`
Expected: PASS (all web_app tests, including the loading-page and 503/loading
tests from Task 3).

- [ ] **Step 6: Commit**

```bash
git add src/pixsage/web/templates/loading.html src/pixsage/web/static/style.css tests/test_web_app.py
git commit -m "feat(web): loading screen template + styles"
```

---

### Task 5: `serve` passes `defer_load=True` (regression test)

**Files:**
- Test: `tests/test_cli_serve.py`

(The production change was made in Task 2 Step 4; this task locks it with a
test.)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli_serve.py`:

```python
def test_serve_defers_model_load(monkeypatch, tmp_path: Path):
    """serve must build the app with defer_load=True so the server answers a
    loading screen instead of blocking ~12s on model load before binding."""
    captured: dict[str, object] = {}

    def fake_build_app(**kwargs):
        captured.update(kwargs)
        raise SystemExit(0)  # bail before uvicorn.run

    monkeypatch.setattr("pixsage.web.app.build_app", fake_build_app)
    runner.invoke(app, ["serve", "--no-open", "--registry", str(tmp_path / "r.json")])

    assert captured.get("defer_load") is True
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python3 -m pytest tests/test_cli_serve.py::test_serve_defers_model_load -v`
Expected: PASS (the Task 2 Step 4 change already passes `defer_load=True`). If it
FAILS, the Task 2 serve edit was missed — fix it there.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cli_serve.py
git commit -m "test(cli): lock serve defers model load"
```

---

### Task 6: Full suite + manual verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `python3 -m pytest tests/ -q --ignore=tests/launcher/test_build_runtime.py --ignore=tests/launcher/test_download_models.py`
Expected: all pass (prior baseline was 269 passed, 42 skipped; this plan adds
~10 tests).

- [ ] **Step 2: Re-stage to the live runtime and time a real launch**

```bash
RT=~/Library/Application\ Support/pixsage
rsync -a --delete /Users/jacksetford/dev/pixsage/src/pixsage/ "$RT/site-packages/pixsage/"
find "$RT/site-packages/pixsage" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
```

Then launch and confirm the browser shows the loading screen immediately and
transitions to search when ready:

```bash
RT=~/Library/Application\ Support/pixsage
HF_HOME="$RT" PYTHONNOUSERSITE=1 PYTHONPATH="$RT/site-packages" \
  "$RT/python/bin/python3" -m pixsage serve
```

Expected: terminal prints `pixsage serve at http://127.0.0.1:8765/` within ~1s;
the opened browser shows the spinner + phase checklist, then reloads into the
search UI once models finish loading (~12s). `Ctrl-C` to stop.

- [ ] **Step 3: No commit** (verification only). Report timings.

---

## Notes for the implementer

- **Why `defer_load` defaults to `False`:** ~6 existing web-test files call
  `build_app(...)` and expect a ready app synchronously. Defaulting to `True`
  would race them. `cli.serve` is the only production caller and opts in.
- **Why reuse `_load_catalog_into_multi`:** it already encapsulates the exact
  per-catalog work (catalog DB, resolver, thumbs, vector store + `service.load()`,
  `multi.add_catalog`) and reads `app.state.embedder`. The background `load_fn`
  sets `app.state.embedder` first, then calls it per entry.
- **Concurrency:** routes never read backend containers while loading (the `/`
  route renders the loading page; the middleware 503s everything else), so the
  thread can populate `app.state` incrementally. `status="ready"` is the final
  store — the publish-then-flag ordering needs no read-path lock under the GIL.
- **`htmx.min.js` is a real one-line minified file** (0 newlines, not empty);
  the loading page deliberately uses inline JS and does not depend on it.
```
