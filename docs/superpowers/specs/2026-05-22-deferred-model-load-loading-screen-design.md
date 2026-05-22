# Deferred model load + loading screen

**Date:** 2026-05-22
**Status:** Approved (brainstorm) — pending implementation plan

## Problem

`pixsage serve` blocks for ~12s (warm) before the web server accepts a single
request. The dominant cost is `embedder.load()` (~11s: torch/transformers
import + deserializing the 1.6 GB SigLIP2 model + device transfer), followed by
the per-catalog vector load (~1.4s after the zero-copy fix in `baea71f`). All
of this happens synchronously inside `build_app()` before `uvicorn.run()`
starts. The launcher opens a browser immediately, so the user stares at a dead
tab spinning on a connection that isn't being accepted yet.

This spec defers that work off the startup path: the server starts serving
near-instantly and a full-page loading screen shows live progress while the
search backend warms up in a background thread, then auto-transitions to the
real search UI.

Scope is `pixsage serve` only (the launch path). `tag`/`embed`/`run` are
unaffected.

## Goals

- `build_app()` returns (and the server accepts requests) in well under a
  second.
- The browser shows a loading screen with phased progress immediately on
  launch, and auto-loads the search UI when the backend is ready.
- No request ever touches half-built backend state.
- Backend load failures surface on the loading screen instead of crashing
  silently or hanging.

## Non-goals

- Speeding up `embedder.load()` itself (the ~11s is mostly unavoidable model
  deserialization).
- The broader UI redesign (the existing `#results`/`.grid` CSS mismatch is a
  separate concern).
- Allowing search or catalog management *during* warmup.

## Architecture

### Backend state machine

A new module `src/pixsage/web/loader.py` defines a `BackendLoader` that owns
readiness state, exposed via `app.state`:

- `status`: `"loading"` → `"ready"` → `"error"`
- `phases`: ordered list of `{label: str, state: "pending"|"active"|"done"}`
- `error`: optional message string (set only when `status == "error"`)

Phases (observable without instrumenting the embedder internals):

1. `"Loading search model…"` — the `embedder.load()` call (~11s)
2. `"Loading catalog vectors…"` — the per-catalog loop (~1.4s)

`BackendLoader` provides:

- A method to mark a phase active/done and to record an error against the
  current phase.
- A `run(load_fn)` entry point that executes `load_fn`, advancing phases,
  catching exceptions into `status="error"`, and setting `status="ready"` as
  the final action on success.
- A thread-safe snapshot accessor for `/status` to read.

### `build_app()` split

`build_app(photo_root, registry_path, embedder_name, *, defer_load=False,
experimental_cluster_labelling=False)` is split into two halves:

**Synchronous half (runs in `build_app`, ~instant):**

- `registry.load()` + `registry.refresh_availability()` + `registry.save()`
- auto-register `photo_root` if given (unchanged logic)
- FastAPI app creation, static mount, route registration
- a fallback `config = _default_config()` so any early render has sane defaults
- a `BackendLoader` in `status="loading"`, stored on `app.state`
- initialize the backend-state containers empty:
  `app.state.multi_search = MultiSearchService()` (empty),
  `catalogs/path_resolvers/thumbs_by_catalog/photoindex_paths = {}`,
  `app.state.embedder = None`

`build_app` then returns immediately.

**Background half (`load_fn`, the slow ~12s):**

Encapsulates today's lines `app.py:90–143`:

- phase 1: `embedder.load(select_device())`, then `app.state.embedder = embedder`
- phase 2: for each enabled+available registry entry, call the existing
  `routes._load_catalog_into_multi(app, entry)` (which already builds `Catalog`,
  `PathResolver`, `ThumbnailCache`, `VectorStore` + `SearchService.load()` and
  calls `multi.add_catalog`, reading `app.state.embedder`), then resolve the
  real `config` from the first catalog's `vocabulary.toml`.

`status="ready"` is set **last**, after phase 2 completes.

**Concurrency model:** during loading, routes never read the backend
containers — the `/` route renders the loading page and the middleware 503s
everything else, both gating on `status`. So the background thread can populate
`app.state.{catalogs,multi_search,...}` incrementally (reusing
`_load_catalog_into_multi` as-is) without locking the read path; readers only
touch those containers once `status == "ready"`, which is flipped as the final
store. Under CPython's GIL the status store is atomic, so this
populate-then-flag ordering is sufficient. `/status` reads a small snapshot
(status/phases/error) guarded by a lock inside `BackendLoader`.

**Threading:** when `defer_load=True`, `build_app` starts a
`threading.Thread(target=loader.run, args=(load_fn,), daemon=True)` before
returning. When `defer_load=False` (the default), it calls `loader.run(load_fn)`
synchronously so the returned app is already `ready`. `cli.serve` passes
`defer_load=True`; the default stays `False` so existing callers/tests get
today's synchronous, already-ready behavior unchanged.

### Routes

- **`GET /status`** (new): returns JSON
  `{status, phases: [{label, state}], error}`. No auth. Cheap. Read by the
  loading page poller.
- **`GET /`**: if `status != "ready"`, render `loading.html` (ignore any
  `?q=`). If `ready`, behave exactly as today.
- **All other routes:** a single HTTP middleware gates everything except `/`,
  `/status`, and `/static/*`. When `status != "ready"` those gated routes return
  **503** with a short "still warming up" JSON body. One middleware (rather than
  editing each route) keeps the gate DRY and prevents any access to half-built
  state — search-dependent routes and registry-mutating routes alike. Catalog
  management is unavailable for the ~12s warmup by design.

### Loading page

`src/pixsage/web/templates/loading.html` — a standalone full page (does not
include `index`'s search form / catalog panel), with minimal inline JS and no
dependency on the empty `htmx.min.js`:

- Centered `pixsage` heading, a CSS spinner, and the phase checklist rendered
  from `/status` (● done / ◑ active / ○ pending).
- Inline JS polls `GET /status` every ~500ms, re-rendering the checklist each
  time. On `status === "ready"` → `location.reload()` (now hits the real search
  page). On `status === "error"` → stop polling, show the error message and a
  hint to check the terminal/logs.
- A small dedicated CSS block appended to `static/style.css` for the loading
  layout + spinner.

## Data flow

```
launcher → browser opens http://127.0.0.1:8765/
  build_app() returns instantly (status=loading), uvicorn serving
  GET /            → loading.html
  [bg thread] embedder.load()  → phase 1 active→done
  [bg thread] catalog loop     → phase 2 active→done
  [bg thread] publish state, status=ready
  loading page poll /status → ready → location.reload()
  GET /            → index.html (real search UI)
```

Error path: background thread raises → loader sets status=error + message →
poll sees error → loading page shows it, stops polling.

## Error handling

- Any exception in `load_fn` is caught by `BackendLoader.run`, recorded against
  the active phase, and surfaced via `status="error"` + `error` message.
- The loading page renders the error and stops polling. The server stays up
  (so `/status` keeps answering); the user is told to check logs.
- `defer_load=False` (tests/sync) propagates nothing differently — the error
  state is observable on `app.state` after `build_app` returns.

## Testing

- **`tests/test_loader.py`** (new): drive `BackendLoader.run` with a fake
  `load_fn`. Assert phase transitions (`pending→active→done`), terminal
  `status="ready"`; and that a raising `load_fn` yields `status="error"` with
  the failing phase and message captured.
- **Route tests** (extend `tests/test_web_app.py`): build a ready app with
  `build_app(..., embedder_name="mock", defer_load=False)`, then drive the
  `loading` state deterministically by setting `app.state.loader.status =
  "loading"` (no thread-timing races). Assert:
  - `/status` JSON shape in `loading` and `ready` states.
  - `/` returns the loading page when not ready, the real search page when
    ready.
  - a gated route (e.g. `/thumb/x/y`) returns 503 while loading; `/status` and
    `/static/...` stay 200.
  - one integration test builds with `defer_load=True` + mock embedder and
    polls `/status` (bounded ~5s wait) to confirm it flips to `ready`.
- **Serve test** (extend `tests/test_cli_serve.py`): monkeypatch
  `pixsage.web.app.build_app` and assert `serve` calls it with
  `defer_load=True`.
- No test downloads models (embedder is mocked, as today).

## Testability seam

`build_app` gains a `defer_load: bool = False` parameter. `cli.serve` passes
`defer_load=True` (background thread). The default `False` preserves today's
synchronous, already-ready behavior for every existing caller and test; loading
state in tests is reproduced by mutating `app.state.loader.status`.

## Files touched

- `src/pixsage/web/loader.py` (new) — `BackendLoader`
- `src/pixsage/web/app.py` — split `build_app`, add `defer_load`, define
  `load_fn`, start thread when deferred
- `src/pixsage/web/routes.py` — `/status` route, `/` loading branch, readiness
  middleware
- `src/pixsage/cli.py` — `serve` passes `defer_load=True`
- `src/pixsage/web/templates/loading.html` (new)
- `src/pixsage/web/static/style.css` — loading layout + spinner block
- `tests/test_loader.py` (new), `tests/test_web_app.py` /
  `tests/test_cli_serve.py` (extend)
```
