# Directory-picker Catalog Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace automatic mounted-volume walking with an explicit in-page folder-browser picker; the server walks only the user-picked subtree and persists every catalog found to the registry.

**Architecture:** Keep `walk_for_photoindex()` but stop calling it at startup. Startup does a cheap path-existence `refresh_availability()` only. A new `GET /catalogs/browse` JSON endpoint lets a vanilla-JS modal navigate the server filesystem; `POST /catalogs/add-scan` walks the chosen directory and bulk-registers finds. `/catalogs/rescan` (volume discovery) becomes `/catalogs/refresh` (availability only).

**Tech Stack:** Python 3.12, FastAPI, Jinja2 templates, pytest + `fastapi.testclient.TestClient`. Spec: `docs/superpowers/specs/2026-05-17-directory-picker-design.md`.

---

## File Structure

- `src/pixsage/registry.py` — add `refresh_availability()`; delete `refresh_from_discovery()` (Task 5).
- `src/pixsage/discovery.py` — add module-level `safe_is_dir()`; delete `list_mounted_roots()` (Task 5).
- `src/pixsage/web/app.py` — startup uses `refresh_availability()`; drop `skip_discovery` param.
- `src/pixsage/web/routes.py` — add `/catalogs/browse`, `/catalogs/add-scan`; replace `/catalogs/rescan` → `/catalogs/refresh`; delete old `/catalogs/add`.
- `src/pixsage/web/templates/_catalogs.html` — folder-browser modal, button text.
- `tests/` — `test_registry.py`, `test_web_catalogs.py` rewritten; `skip_discovery=True` kwarg removed everywhere.

Each commit leaves the full suite green.

---

### Task 1: Registry — add `refresh_availability()`

**Files:**
- Modify: `src/pixsage/registry.py` (after `mark_available`, before `refresh_from_discovery`, ~line 141)
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_registry.py`:

```python
def test_refresh_availability_marks_existing(tmp_path):
    from pixsage.registry import Registry
    pi = tmp_path / "Sony" / ".photoindex"
    pi.mkdir(parents=True)
    reg = Registry(tmp_path / "catalogs.json")
    reg.load()
    reg.add(photoindex_path=str(pi.resolve()), label="Sony",
            image_embedder_signature="i", caption_embedder_signature="c")
    reg.refresh_availability()
    e = list(reg.entries())[0]
    assert e.available is True
    assert e.last_seen  # bumped


def test_refresh_availability_marks_offline_and_adds_nothing(tmp_path):
    from pixsage.registry import Registry
    reg = Registry(tmp_path / "catalogs.json")
    reg.load()
    reg.add(photoindex_path="/Volumes/NotMounted/.photoindex", label="Gone",
            image_embedder_signature="i", caption_embedder_signature="c")
    reg.refresh_availability()
    entries = list(reg.entries())
    assert len(entries) == 1          # never adds
    assert entries[0].available is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_registry.py -k refresh_availability -v`
Expected: FAIL — `AttributeError: 'Registry' object has no attribute 'refresh_availability'`

- [ ] **Step 3: Implement `refresh_availability`**

Insert in `src/pixsage/registry.py` immediately before `def refresh_from_discovery` (~line 142):

```python
    def refresh_availability(self) -> None:
        """Re-check whether each registered catalog's path currently exists.

        Pure existence check — no filesystem walk, never adds or removes
        entries. Called at startup and by the availability-refresh route so
        a (re)plugged drive flips online/offline without crawling disks.
        """
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        for e in self._entries:
            e.available = Path(e.photoindex_path).exists()
            if e.available:
                e.last_seen = now
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_registry.py -k refresh_availability -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/registry.py tests/test_registry.py
git commit -m "feat(registry): add refresh_availability (no-walk existence check)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Startup uses `refresh_availability`; drop `skip_discovery`

`skip_discovery=True` is passed in: `tests/test_web_app.py`, `test_web_search.py`, `test_web_catalogs.py`, `test_web_app_multi.py`, `test_serve_path_translation.py`, `test_web_clusters.py`. All must lose the kwarg in this task so the suite stays green.

**Files:**
- Modify: `src/pixsage/web/app.py:50-93`
- Modify: the 6 test files above (mechanical kwarg removal)
- Test: `tests/test_web_app.py` (new no-walk assertion)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_web_app.py`:

```python
def test_startup_does_not_walk_filesystem(tmp_path, monkeypatch):
    """build_app() must not trigger a recursive discovery walk at startup."""
    import pixsage.discovery as disc
    from pixsage.web.app import build_app

    called = {"n": 0}
    monkeypatch.setattr(disc, "walk_for_photoindex",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1) or [])
    build_app(registry_path=tmp_path / "catalogs.json", embedder_name="mock")
    assert called["n"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web_app.py::test_startup_does_not_walk_filesystem -v`
Expected: FAIL — `assert 1 == 0` (startup currently walks).

- [ ] **Step 3: Edit `build_app`**

In `src/pixsage/web/app.py`, change the signature — delete the `skip_discovery` parameter and its docstring line:

Replace lines 50-57 (the `def build_app(...)` through `) -> FastAPI:`) so the params are exactly:

```python
def build_app(
    photo_root: Path | None = None,
    registry_path: Path | None = None,
    embedder_name: str = "siglip2",
    *,
    experimental_cluster_labelling: bool = False,
) -> FastAPI:
```

Delete the docstring lines describing `skip_discovery:` (the two lines starting `skip_discovery: If True...` / `Useful in tests...`).

Replace the discovery block (currently lines ~86-93):

```python
    # Discovery + availability reconciliation.
    if not skip_discovery:
        from pixsage.discovery import list_mounted_roots, walk_for_photoindex
        discovered = walk_for_photoindex(list_mounted_roots())
        registry.refresh_from_discovery(discovered)
    else:
        registry.refresh_from_discovery(discovered_paths=[])
    registry.save()
```

with:

```python
    # No startup discovery walk — catalogs enter the registry only via the
    # folder-browser picker (POST /catalogs/add-scan) or an explicit
    # photo_root arg. Startup only re-checks which registered paths exist.
    registry.refresh_availability()
    registry.save()
```

- [ ] **Step 4: Remove `skip_discovery=True` from all test call sites**

Run this to find every occurrence:

```bash
grep -rln "skip_discovery" tests/
```

In each listed file, delete the `, skip_discovery=True` argument (or `skip_discovery=True,` / standalone) from every `build_app(...)` call. Do not change anything else. Verify none remain:

```bash
grep -rn "skip_discovery" src/ tests/
```
Expected: no output.

- [ ] **Step 5: Run the affected suites**

Run: `pytest tests/test_web_app.py tests/test_web_search.py tests/test_web_app_multi.py tests/test_serve_path_translation.py tests/test_web_clusters.py -q`
Expected: all PASS (note: `test_web_catalogs.py` is intentionally rewritten in Task 5 and may have failing rescan tests until then — exclude it here).

- [ ] **Step 6: Commit**

```bash
git add src/pixsage/web/app.py tests/
git commit -m "refactor(web): no startup discovery walk; drop skip_discovery

Startup now only runs registry.refresh_availability(). Removes the
skip_discovery test escape hatch (default behavior is now no-walk).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `safe_is_dir` helper + `GET /catalogs/browse`

**Files:**
- Modify: `src/pixsage/discovery.py` (add module-level helper near top, after `SKIP_DIRS`)
- Modify: `src/pixsage/web/routes.py` (new route near other `/catalogs/*` routes, ~after line 161)
- Test: `tests/test_web_catalogs.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_web_catalogs.py`:

```python
def test_browse_lists_child_dirs_and_photoindex_hint(tmp_path):
    from pixsage.web.app import build_app
    (tmp_path / "Sony").mkdir()
    (tmp_path / "Sony" / ".photoindex").mkdir()
    (tmp_path / "Empty").mkdir()
    app = build_app(registry_path=tmp_path / "catalogs.json", embedder_name="mock")
    with TestClient(app) as client:
        r = client.get("/catalogs/browse", params={"path": str(tmp_path)})
        assert r.status_code == 200
        body = r.json()
        assert body["path"] == str(tmp_path.resolve())
        names = {e["name"]: e for e in body["entries"]}
        assert names["Sony"]["has_photoindex"] is True
        assert names["Empty"]["has_photoindex"] is False
        assert body["parent"] == str(tmp_path.resolve().parent)


def test_browse_rejects_bad_path(tmp_path):
    from pixsage.web.app import build_app
    app = build_app(registry_path=tmp_path / "catalogs.json", embedder_name="mock")
    with TestClient(app) as client:
        r = client.get("/catalogs/browse", params={"path": str(tmp_path / "nope")})
        assert r.status_code == 400
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_web_catalogs.py -k browse -v`
Expected: FAIL — 404 (route not defined).

- [ ] **Step 3: Add `safe_is_dir` to `discovery.py`**

In `src/pixsage/discovery.py`, immediately after the `SKIP_DIRS = frozenset({...})` block (~line 28), add:

```python
def safe_is_dir(p: "Path") -> bool:
    """Path.is_dir() that treats un-stattable paths as non-dirs.

    Path.is_dir() swallows generic OSError but NOT PermissionError
    (EACCES) — SIP-protected files under a system volume would otherwise
    raise. Used by the walker and the folder-browser endpoint.
    """
    try:
        return p.is_dir()
    except OSError:
        return False
```

(`Path` is already imported at the top of `discovery.py`.)

- [ ] **Step 4: Add the `/catalogs/browse` route**

In `src/pixsage/web/routes.py`, after the `add_catalog` route (after line 161, before `@app.post("/catalogs/{catalog_id}/remove")`), add:

```python
    @app.get("/catalogs/browse")
    def browse_dirs(path: str | None = None) -> dict:
        from pixsage.discovery import safe_is_dir

        base = Path(path).expanduser() if path else Path.home()
        try:
            base = base.resolve()
        except OSError:
            raise HTTPException(status_code=400, detail=f"bad path: {path}")
        if not (base.exists() and base.is_dir()):
            raise HTTPException(status_code=400, detail=f"not a directory: {base}")

        entries = []
        try:
            children = sorted(base.iterdir(), key=lambda c: c.name.lower())
        except OSError:
            children = []
        for c in children:
            if c.name.startswith(".") or not safe_is_dir(c):
                continue
            entries.append({
                "name": c.name,
                "path": str(c),
                "has_photoindex": (c / ".photoindex").exists(),
            })

        roots = [{"name": "Home", "path": str(Path.home())}]
        volumes = Path("/Volumes")
        if volumes.is_dir():
            try:
                for v in sorted(volumes.iterdir(), key=lambda c: c.name.lower()):
                    if safe_is_dir(v):
                        roots.append({"name": v.name, "path": str(v)})
            except OSError:
                pass

        parent = str(base.parent) if base.parent != base else None
        return {"path": str(base), "parent": parent, "entries": entries, "roots": roots}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_web_catalogs.py -k browse -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add src/pixsage/discovery.py src/pixsage/web/routes.py tests/test_web_catalogs.py
git commit -m "feat(web): GET /catalogs/browse filesystem listing for picker

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `POST /catalogs/add-scan` (walk picked dir, bulk-register)

**Files:**
- Modify: `src/pixsage/web/routes.py` (new route after `browse_dirs`)
- Test: `tests/test_web_catalogs.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_web_catalogs.py`:

```python
def test_add_scan_registers_nested_catalogs(tmp_path):
    from pixsage.web.app import build_app
    a = tmp_path / "drive" / "ShootA"
    b = tmp_path / "drive" / "ShootB"
    _make_catalog(a / ".photoindex", photo_root=a)
    _make_catalog(b / ".photoindex", photo_root=b)
    app = build_app(registry_path=tmp_path / "catalogs.json", embedder_name="mock")
    with TestClient(app) as client:
        r = client.post("/catalogs/add-scan",
                         data={"path": str(tmp_path / "drive")},
                         follow_redirects=False)
        assert r.status_code in (302, 303)
        reg = Registry(tmp_path / "catalogs.json")
        reg.load()
        labels = sorted(e.label for e in reg.entries())
        assert labels == ["ShootA", "ShootB"]


def test_add_scan_empty_subtree_adds_nothing(tmp_path):
    from pixsage.web.app import build_app
    (tmp_path / "plain").mkdir()
    app = build_app(registry_path=tmp_path / "catalogs.json", embedder_name="mock")
    with TestClient(app) as client:
        r = client.post("/catalogs/add-scan", data={"path": str(tmp_path / "plain")},
                         follow_redirects=False)
        assert r.status_code in (302, 303)
        reg = Registry(tmp_path / "catalogs.json")
        reg.load()
        assert list(reg.entries()) == []


def test_add_scan_dedupes_already_registered(tmp_path):
    from pixsage.web.app import build_app
    s = tmp_path / "Sony"
    _make_catalog(s / ".photoindex", photo_root=s)
    app = build_app(registry_path=tmp_path / "catalogs.json", embedder_name="mock")
    with TestClient(app) as client:
        client.post("/catalogs/add-scan", data={"path": str(tmp_path)})
        client.post("/catalogs/add-scan", data={"path": str(tmp_path)})
        reg = Registry(tmp_path / "catalogs.json")
        reg.load()
        assert len(list(reg.entries())) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_web_catalogs.py -k add_scan -v`
Expected: FAIL — 404 (route not defined).

- [ ] **Step 3: Add the `/catalogs/add-scan` route**

In `src/pixsage/web/routes.py`, immediately after the `browse_dirs` route, add:

```python
    @app.post("/catalogs/add-scan")
    def add_scan(path: str = Form(...)) -> RedirectResponse:
        from pixsage import discovery
        from pixsage.registry import derive_signatures

        registry = app.state.registry
        root = Path(path).expanduser()
        if not (root.exists() and root.is_dir()):
            raise HTTPException(status_code=400, detail=f"not a directory: {root}")

        found = discovery.walk_for_photoindex([root])
        added = 0
        for pi in found:
            pi = Path(pi).resolve()
            if registry.find_by_photoindex_path(str(pi)) is not None:
                continue
            img_sig, cap_sig = derive_signatures(pi)
            entry = registry.add(
                photoindex_path=str(pi),
                label=pi.parent.name,
                image_embedder_signature=img_sig,
                caption_embedder_signature=cap_sig,
            )
            entry.available = True
            _load_catalog_into_multi(app, entry)
            added += 1
        registry.save()
        return RedirectResponse(url="/", status_code=303)
```

(`Form`, `RedirectResponse`, `HTTPException`, `Path` are already imported in `routes.py`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_web_catalogs.py -k add_scan -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/web/routes.py tests/test_web_catalogs.py
git commit -m "feat(web): POST /catalogs/add-scan bulk-registers picked subtree

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Replace `/catalogs/rescan`→`/catalogs/refresh`; delete dead discovery code

**Files:**
- Modify: `src/pixsage/web/routes.py:205-232` (replace `rescan_catalogs`); delete `add_catalog` (`/catalogs/add`, lines 131-161)
- Modify: `src/pixsage/discovery.py` (delete `list_mounted_roots`)
- Modify: `src/pixsage/registry.py` (delete `refresh_from_discovery`)
- Test: `tests/test_web_catalogs.py` (rewrite rescan + old-add tests), `tests/test_registry.py` (drop `refresh_from_discovery` tests)

- [ ] **Step 1: Rewrite the affected tests (failing first)**

In `tests/test_web_catalogs.py`:
- Delete `test_add_catalog_with_photoindex_path_directly_uses_parent_label` (the `/catalogs/add` route is being removed; `add-scan` already covers labelling via `test_add_scan_registers_nested_catalogs`).
- Delete `test_rescan_*` tests that `monkeypatch` `list_mounted_roots` (the two around lines ~200-256) and replace with:

```python
def test_refresh_marks_offline_then_back(tmp_path):
    from pixsage.web.app import build_app
    s = tmp_path / "Sony"
    _make_catalog(s / ".photoindex", photo_root=s)
    app = build_app(registry_path=tmp_path / "catalogs.json", embedder_name="mock")
    with TestClient(app) as client:
        client.post("/catalogs/add-scan", data={"path": str(tmp_path)})
        assert len(app.state.multi_search.catalog_ids()) == 1

        import gc
        for cat in list(app.state.catalogs.values()):
            cat.close()
        gc.collect()
        offline = tmp_path / "Sony.photoindex.offline"
        (s / ".photoindex").rename(offline)
        client.post("/catalogs/refresh")
        assert len(app.state.multi_search.catalog_ids()) == 0

        offline.rename(s / ".photoindex")
        client.post("/catalogs/refresh")
        assert len(app.state.multi_search.catalog_ids()) == 1
```

In `tests/test_registry.py`: delete the tests that call `reg.refresh_from_discovery(...)` (the cases around lines ~205-272). Coverage is replaced by Task 1's `refresh_availability` tests + Task 4's add-scan dedupe test.

Run: `pytest tests/test_web_catalogs.py -k refresh_marks_offline -v`
Expected: FAIL — 404 (`/catalogs/refresh` not defined).

- [ ] **Step 2: Replace the rescan route with refresh**

In `src/pixsage/web/routes.py`, replace the entire `@app.post("/catalogs/rescan")` block (lines 205-232) with:

```python
    @app.post("/catalogs/refresh")
    def refresh_catalogs() -> RedirectResponse:
        registry = app.state.registry
        multi = app.state.multi_search
        registry.refresh_availability()
        registry.save()

        # Reconcile loaded state vs target (enabled + available).
        loaded_ids = set(multi.catalog_ids())
        for entry in registry.entries():
            should = entry.enabled and entry.available
            is_loaded = entry.id in loaded_ids
            if should and not is_loaded:
                _load_catalog_into_multi(app, entry)
            elif is_loaded and not should:
                multi.remove_catalog(entry.id)
                app.state.catalogs.pop(entry.id, None)
                app.state.path_resolvers.pop(entry.id, None)
                app.state.thumbs_by_catalog.pop(entry.id, None)
                app.state.photoindex_paths.pop(entry.id, None)
        return RedirectResponse(url="/", status_code=303)
```

- [ ] **Step 3: Delete the old `/catalogs/add` route**

In `src/pixsage/web/routes.py`, delete the entire `@app.post("/catalogs/add")` block (lines 131-161, `def add_catalog` through its `return RedirectResponse`). The folder-browser modal posts to `/catalogs/add-scan` instead.

- [ ] **Step 4: Delete `list_mounted_roots` and `refresh_from_discovery`**

In `src/pixsage/discovery.py`: delete the entire `def list_mounted_roots() -> list[Path]:` function (lines ~30-76) including its docstring. Keep `SKIP_DIRS`, `safe_is_dir`, `walk_for_photoindex`.

In `src/pixsage/registry.py`: delete the entire `def refresh_from_discovery(self, discovered_paths: list[Path]) -> None:` method (lines ~142-175). Keep `refresh_availability`, `add`, `derive_signatures`.

Verify nothing references the deleted symbols:

```bash
grep -rn "list_mounted_roots\|refresh_from_discovery\|catalogs/rescan\|catalogs/add\"" src/ tests/
```
Expected: no output.

- [ ] **Step 5: Run full suite to verify green**

Run: `pytest -q`
Expected: all PASS. (If `tests/test_discovery.py` references `list_mounted_roots`, it does not — it only imports `walk_for_photoindex`; no change needed there.)

- [ ] **Step 6: Commit**

```bash
git add src/pixsage tests/
git commit -m "refactor: drop volume discovery; /catalogs/rescan -> /catalogs/refresh

Removes list_mounted_roots, refresh_from_discovery, and the paste-path
/catalogs/add route. Availability refresh replaces volume rescan.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Folder-browser modal in `_catalogs.html`

**Files:**
- Modify: `src/pixsage/web/templates/_catalogs.html`
- Test: `tests/test_web_catalogs.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_web_catalogs.py`:

```python
def test_index_has_folder_browser_modal(tmp_path):
    from pixsage.web.app import build_app
    app = build_app(registry_path=tmp_path / "catalogs.json", embedder_name="mock")
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert 'id="catalog-browser"' in r.text
        assert "/catalogs/add-scan" in r.text
        assert "/catalogs/refresh" in r.text
        assert "/catalogs/rescan" not in r.text
        assert "/catalogs/add\"" not in r.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web_catalogs.py::test_index_has_folder_browser_modal -v`
Expected: FAIL — old template still has `/catalogs/rescan` and `/catalogs/add`.

- [ ] **Step 3: Replace the actions block in `_catalogs.html`**

Replace the empty-state `<p>` (line 27) text and the entire `<div class="catalogs-actions">…</div>` block (lines 30-38) with:

```html
  {% else %}
    <p class="empty-state">No catalogs yet. Click <strong>Add catalog…</strong> and pick a folder (or a whole drive) that contains indexed <code>.photoindex/</code> catalogs.</p>
  {% endif %}

  <div class="catalogs-actions">
    <button type="button" onclick="document.getElementById('catalog-browser').showModal()">Add catalog&hellip;</button>
    <form method="post" action="/catalogs/refresh" style="display:inline">
      <button type="submit">Refresh availability</button>
    </form>
  </div>

  <dialog id="catalog-browser">
    <h3>Pick a folder to scan for catalogs</h3>
    <p class="cb-current" id="cb-current"></p>
    <ul id="cb-list" class="cb-list"></ul>
    <form method="post" action="/catalogs/add-scan">
      <input type="hidden" name="path" id="cb-path">
      <button type="submit">Scan this folder &amp; add catalogs</button>
      <button type="button" onclick="document.getElementById('catalog-browser').close()">Cancel</button>
    </form>
  </dialog>

  <script>
  (function () {
    var dlg = document.getElementById('catalog-browser');
    if (!dlg) return;
    function render(d) {
      document.getElementById('cb-current').textContent = d.path;
      document.getElementById('cb-path').value = d.path;
      var ul = document.getElementById('cb-list');
      ul.innerHTML = '';
      d.roots.forEach(function (r) {
        var li = document.createElement('li');
        var a = document.createElement('a');
        a.href = '#'; a.textContent = '⌂ ' + r.name;
        a.onclick = function (e) { e.preventDefault(); load(r.path); };
        li.appendChild(a); ul.appendChild(li);
      });
      if (d.parent) {
        var up = document.createElement('li');
        var ua = document.createElement('a');
        ua.href = '#'; ua.textContent = '.. (up)';
        ua.onclick = function (e) { e.preventDefault(); load(d.parent); };
        up.appendChild(ua); ul.appendChild(up);
      }
      d.entries.forEach(function (en) {
        var li = document.createElement('li');
        var a = document.createElement('a');
        a.href = '#';
        a.textContent = en.name + (en.has_photoindex ? '  ✓ catalog' : '');
        a.onclick = function (e) { e.preventDefault(); load(en.path); };
        li.appendChild(a); ul.appendChild(li);
      });
    }
    function load(p) {
      fetch('/catalogs/browse?path=' + encodeURIComponent(p))
        .then(function (r) { return r.json(); })
        .then(render);
    }
    dlg.addEventListener('close', function () {});
    document.querySelector('.catalogs-actions button')
      .addEventListener('click', function () { load(''); });
  })();
  </script>
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_web_catalogs.py::test_index_has_folder_browser_modal -v`
Expected: PASS

- [ ] **Step 5: Manual smoke check**

Run: `PYTHONPATH=src python -m pixsage serve --no-open` in a temp dir with a `.photoindex/` somewhere under `$HOME`; open `http://127.0.0.1:8765/`, click **Add catalog…**, navigate, click **Scan this folder**, confirm the catalog appears in the panel. Ctrl-C to stop.
Expected: modal opens, navigation works, catalog registered.

- [ ] **Step 6: Commit**

```bash
git add src/pixsage/web/templates/_catalogs.html tests/test_web_catalogs.py
git commit -m "feat(web): folder-browser modal replaces paste-path + rescan button

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Full verification

- [ ] **Step 1: Run the whole suite**

Run: `pytest -q`
Expected: all PASS, no errors.

- [ ] **Step 2: Confirm dead code is gone**

Run: `grep -rn "list_mounted_roots\|refresh_from_discovery\|skip_discovery\|catalogs/rescan" src/ tests/`
Expected: no output.

- [ ] **Step 3: Confirm the launcher still boots (regression)**

Run: `PYTHONPATH=src python -m pixsage serve --no-open` ; in another shell `curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8765/` → expect `200`; Ctrl-C.

- [ ] **Step 4: Final commit (if any uncommitted changes)**

```bash
git add -A && git commit -m "chore: directory-picker discovery complete

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>" || echo "nothing to commit"
```

---

## Self-Review

**Spec coverage:**
- Spec §1 (remove auto-discovery, keep walker) → Task 2 (startup) + Task 5 (delete `list_mounted_roots`). ✓
- Spec §2 (split reconciliation) → Task 1 (`refresh_availability`) + Task 5 (delete `refresh_from_discovery`); bulk-add → Task 4. ✓
- Spec §3 (browse endpoint, add-scan, modal, rescan→refresh, fold `/catalogs/add`) → Tasks 3, 4, 5, 6. ✓
- Spec §4 (CLI back-compat) → unchanged; `photo_root` path in `build_app` untouched by Task 2 edit. ✓
- Spec §5 (test migration) → Task 2 Step 4, Task 5 Step 1. ✓
- Spec testing strategy → Tasks 1,3,4,5,6 tests + Task 7. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. ✓

**Type/name consistency:** `refresh_availability()` (Task 1) used identically in Tasks 2, 5. `safe_is_dir` defined Task 3, used Tasks 3. `/catalogs/add-scan`, `/catalogs/browse`, `/catalogs/refresh` consistent across Tasks 3-6. `_load_catalog_into_multi(app, entry)` signature matches existing `routes.py:412`. ✓
