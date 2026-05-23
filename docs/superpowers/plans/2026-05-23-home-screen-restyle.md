# Home-Screen Restyle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restyle the pixsage home page into a search-first layout, replace the inline `<details>` catalog panel with a `<dialog>` modal opened from a one-line collapsed strip, and fix the results-grid CSS/markup mismatch.

**Architecture:** Presentation-layer change only — no Python, no route, no Jinja context changes. Three Jinja templates edited (`index.html`, `_catalogs.html`, `_results.html`) and one stylesheet rewritten (`style.css`). Tests use FastAPI's `TestClient` against the existing `build_app` factory; the new tests assert on rendered HTML, not on visual styling.

**Tech Stack:** Jinja2 templates, plain CSS (no preprocessor), native HTML `<dialog>`, FastAPI `TestClient` for tests.

---

## File map

| Path | Action | Responsibility |
| --- | --- | --- |
| `src/pixsage/web/templates/_results.html` | Modify | Drop the wrapping `<div class="grid">` so cards tile under the existing `#results` grid rule. |
| `src/pixsage/web/templates/_catalogs.html` | Modify (major) | Add a collapsed `.catalogs-strip` block; convert the outer `<details>` to `<dialog class="catalogs-modal">`; add a small inline JS handler to open the modal and close on backdrop click. |
| `src/pixsage/web/templates/index.html` | Modify | Move the Caption ⇄ Visual slider out of the search-input row into a sibling `<div class="weight">` below it. |
| `src/pixsage/web/static/style.css` | Modify (major) | Update `body` bg; add all the new selectors enumerated in the design spec (strip, search form, modal frame, list rows, folder picker). Leave cluster/photo/loading rules untouched. |
| `tests/test_web_results_grid.py` | Create | Lock the `_results.html` fix — assert the rendered home page (with hits) does **not** contain `class="grid"`. |
| `tests/test_web_catalogs.py` | Modify | Append two tests: (a) rendered home contains the collapsed `.catalogs-strip`; (b) rendered home contains `<dialog class="catalogs-modal" id="catalogs-modal">`. Existing tests must keep passing. |

No new Python source files. No changes to `routes.py`, `app.py`, `loader.py`, `registry.py`.

---

## Task 1: Lock the results-grid bug with a failing test, then fix it

**Files:**
- Create: `tests/test_web_results_grid.py`
- Modify: `src/pixsage/web/templates/_results.html`

- [ ] **Step 1: Write the failing test**

Create `tests/test_web_results_grid.py`:

```python
from __future__ import annotations
from pathlib import Path

from fastapi.testclient import TestClient


def test_results_partial_does_not_emit_grid_div(tmp_path: Path) -> None:
    """The `_results.html` partial must not wrap cards in `class="grid"`.

    `style.css` grids the parent `<section id="results">`; a nested
    `.grid` div would break the layout (the parent grid sees one full-
    width child instead of the cards).
    """
    from pixsage.web.app import build_app

    registry_path = tmp_path / "catalogs.json"
    app = build_app(registry_path=registry_path, embedder_name="mock")
    with TestClient(app) as client:
        # Hit the home page with a query so the results partial is rendered.
        # With no catalogs and mock embedder there are no hits, but the
        # partial still renders an (empty) container — what we care about
        # is that no wrapping `class="grid"` is emitted anywhere in the
        # rendered page.
        r = client.get("/?q=anything")
        assert r.status_code == 200
        assert 'class="grid"' not in r.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web_results_grid.py -v`
Expected: FAIL with `assert 'class="grid"' not in r.text` because `_results.html` currently emits `<div class="grid">`.

- [ ] **Step 3: Read the current `_results.html` so you know exactly what to remove**

Run: `cat src/pixsage/web/templates/_results.html`

Expected contents (current):

```jinja
{% if hits %}
  <p class="results-meta">{{ hits|length }} result{{ 's' if hits|length != 1 }}</p>
  <div class="grid">
    {% for hit in hits %}
      {% include "_card.html" %}
    {% endfor %}
  </div>
{% else %}
  <p class="no-results">No matches.</p>
{% endif %}
```

(If the file looks different in any non-trivial way, stop and ask — the spec assumes this shape.)

- [ ] **Step 4: Edit `_results.html` to drop the wrapping div**

Replace the whole file with:

```jinja
{% if hits %}
  <p class="results-meta">{{ hits|length }} result{{ 's' if hits|length != 1 }}</p>
  {% for hit in hits %}
    {% include "_card.html" %}
  {% endfor %}
{% else %}
  <p class="no-results">No matches.</p>
{% endif %}
```

The cards now render as direct children of `<section id="results">` in `index.html`, which already has `display: grid` in `style.css`.

- [ ] **Step 5: Run the new test + the existing web suite to verify nothing else broke**

Run: `pytest tests/test_web_results_grid.py tests/test_web_app.py tests/test_web_app_multi.py tests/test_web_search.py tests/test_web_catalogs.py tests/test_web_clusters.py tests/test_web_thumbs.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tests/test_web_results_grid.py src/pixsage/web/templates/_results.html
git commit -m "fix(web): drop wrapping .grid div from _results.html

#results already has the grid rule; the nested .grid was stacking
cards full-width."
```

---

## Task 2: Add the collapsed catalogs-strip

**Files:**
- Modify: `src/pixsage/web/templates/_catalogs.html`
- Modify: `tests/test_web_catalogs.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_web_catalogs.py` (at the end of the file):

```python
def test_home_renders_collapsed_catalogs_strip(tmp_path: Path) -> None:
    """The home page renders a one-line `.catalogs-strip` summary
    above the search form. The full management UI lives in a modal
    that is opened from this strip's `Manage ▸` button.
    """
    from pixsage.web.app import build_app

    sony = tmp_path / "Sony"
    _make_catalog(sony / ".photoindex", photo_root=sony)

    registry_path = tmp_path / "catalogs.json"
    reg = Registry(registry_path)
    reg.load()
    reg.add(photoindex_path=str((sony / ".photoindex").resolve()),
            label="Sony",
            image_embedder_signature="siglip2-so400m-patch14-384@v1",
            caption_embedder_signature="minilm-L6-v2@v2")
    reg.save()

    app = build_app(registry_path=registry_path, embedder_name="mock")
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert 'class="catalogs-strip"' in r.text
        # The strip surfaces a Manage affordance.
        assert "Manage" in r.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web_catalogs.py::test_home_renders_collapsed_catalogs_strip -v`
Expected: FAIL with `assert 'class="catalogs-strip"' in r.text` — the strip doesn't exist yet.

- [ ] **Step 3: Read the current `_catalogs.html`**

Run: `cat src/pixsage/web/templates/_catalogs.html`

Note the existing structure: a `<details class="catalogs-panel">` wrapping a `<summary>`, then the catalog list / actions / nested `<dialog id="catalog-browser">` and its `<script>`.

- [ ] **Step 4: Add the strip at the top of `_catalogs.html`**

Edit `src/pixsage/web/templates/_catalogs.html`. **Add this block at the very top** (before the existing `{% set entries = ... %}` line):

```jinja
{# Collapsed one-line summary shown on the home page. Clicking Manage
   opens the catalogs modal that holds the full list / actions. #}
{% set _entries = registry.entries() | list %}
{% set _first_available = (_entries | selectattr('available') | list | first) %}
{% set _count = _entries | length %}
<div class="catalogs-strip">
  <span class="cs-count">{{ _count }} catalog{{ '' if _count == 1 else 's' }}</span>
  {% if _first_available %}
    <span class="cs-sep">·</span>
    <span class="cs-path">{{ _first_available.photoindex_path }}</span>
  {% elif _count > 0 %}
    <span class="cs-sep">·</span>
    <span class="cs-path">offline</span>
  {% endif %}
  <button type="button" class="cs-manage"
          onclick="document.getElementById('catalogs-modal').showModal()">
    {% if _count == 0 %}Add one ▸{% else %}Manage ▸{% endif %}
  </button>
</div>
```

(The rest of `_catalogs.html` — the existing `<details>` panel — stays in place for now. Task 3 converts that to the modal.)

- [ ] **Step 5: Run the new test + the catalog suite to verify**

Run: `pytest tests/test_web_catalogs.py -v`
Expected: all pass, including the new `test_home_renders_collapsed_catalogs_strip`.

- [ ] **Step 6: Commit**

```bash
git add src/pixsage/web/templates/_catalogs.html tests/test_web_catalogs.py
git commit -m "feat(web): collapsed catalogs strip above search form

One-line summary (count + first available path + Manage button).
Manage will open the catalogs modal once it exists."
```

---

## Task 3: Convert the catalogs panel from `<details>` to `<dialog>` modal

**Files:**
- Modify: `src/pixsage/web/templates/_catalogs.html`
- Modify: `tests/test_web_catalogs.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_web_catalogs.py`:

```python
def test_home_renders_catalogs_modal_dialog(tmp_path: Path) -> None:
    """The catalog management UI must be rendered as a `<dialog>` with
    id `catalogs-modal`, not the old `<details>` panel. The inner
    form actions (rename / toggle / remove / refresh / add-scan) and
    the nested folder-picker `<dialog id="catalog-browser">` must
    still be present.
    """
    from pixsage.web.app import build_app

    sony = tmp_path / "Sony"
    _make_catalog(sony / ".photoindex", photo_root=sony)

    registry_path = tmp_path / "catalogs.json"
    reg = Registry(registry_path)
    reg.load()
    reg.add(photoindex_path=str((sony / ".photoindex").resolve()),
            label="Sony",
            image_embedder_signature="siglip2-so400m-patch14-384@v1",
            caption_embedder_signature="minilm-L6-v2@v2")
    reg.save()

    app = build_app(registry_path=registry_path, embedder_name="mock")
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert '<dialog class="catalogs-modal" id="catalogs-modal">' in r.text
        # Old wrapper must be gone.
        assert 'class="catalogs-panel"' not in r.text
        # Form actions still present.
        assert "/catalogs/refresh" in r.text
        assert "/catalogs/add-scan" in r.text
        # Folder-picker dialog still present.
        assert 'id="catalog-browser"' in r.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web_catalogs.py::test_home_renders_catalogs_modal_dialog -v`
Expected: FAIL — the outer element is still `<details class="catalogs-panel">`.

- [ ] **Step 3: Replace the outer wrapper in `_catalogs.html`**

In `src/pixsage/web/templates/_catalogs.html`:

(a) **Find** the line:
```jinja
<details class="catalogs-panel" {% if not has_any_enabled %}open{% endif %}>
  <summary>Catalogs ({{ entries | length }})</summary>
```

(b) **Replace** it with:
```jinja
<dialog class="catalogs-modal" id="catalogs-modal">
  <header class="catalogs-modal-header">
    <h2>Catalogs ({{ entries | length }})</h2>
    <button type="button" class="modal-close" aria-label="Close"
            onclick="document.getElementById('catalogs-modal').close()">×</button>
  </header>
  <div class="catalogs-modal-body">
```

(c) **Find** the closing `</details>` at the bottom of the file and **replace** it with:
```jinja
  </div>
</dialog>
```

(d) The `{% set has_any_enabled = ... %}` line that was used to auto-open the `<details>` can be deleted — it's unused now. Search the file for `has_any_enabled` and remove the `{% set %}` line.

(e) The existing inline `<script>` that wires the **Add catalog…** button to the folder picker stays as-is. **Add** this small handler immediately after it, just before the final `</dialog>`:

```jinja
  <script>
  (function () {
    var modal = document.getElementById('catalogs-modal');
    if (!modal) return;
    // Close when the user clicks the backdrop (outside the modal box).
    modal.addEventListener('click', function (e) {
      if (e.target === modal) modal.close();
    });
  })();
  </script>
```

- [ ] **Step 4: Run the new test + the existing catalog tests**

Run: `pytest tests/test_web_catalogs.py -v`
Expected: all pass — the new modal test plus all existing `test_panel_*` tests.

- [ ] **Step 5: Run the full web test surface to verify nothing else broke**

Run: `pytest tests/test_web_app.py tests/test_web_app_multi.py tests/test_web_search.py tests/test_web_catalogs.py tests/test_web_clusters.py tests/test_web_thumbs.py tests/test_web_results_grid.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/pixsage/web/templates/_catalogs.html tests/test_web_catalogs.py
git commit -m "refactor(web): catalogs panel becomes a <dialog> modal

Opened from the .catalogs-strip's Manage button. Inner form actions
and the nested folder-picker dialog are preserved untouched."
```

---

## Task 4: Move the Caption ⇄ Visual slider to its own row

**Files:**
- Modify: `src/pixsage/web/templates/index.html`
- Modify: `tests/test_web_catalogs.py` (or a new test file — choose existing for cohesion)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_web_catalogs.py`:

```python
def test_home_search_slider_lives_in_weight_row(tmp_path: Path) -> None:
    """The Caption ⇄ Visual slider sits in a sibling `.weight` row
    below the search input, not inline with the input.
    """
    from pixsage.web.app import build_app

    registry_path = tmp_path / "catalogs.json"
    app = build_app(registry_path=registry_path, embedder_name="mock")
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert 'class="weight"' in r.text
        # The slider's `name` attribute is unchanged.
        assert 'name="image_weight"' in r.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web_catalogs.py::test_home_search_slider_lives_in_weight_row -v`
Expected: FAIL — no `.weight` element yet.

- [ ] **Step 3: Read the current `index.html` search form**

Run: `cat src/pixsage/web/templates/index.html`

Note the current structure inside the `{% else %}` branch (no `similar_to`):

```jinja
<form id="search-form" method="get" action="/">
  <input type="search" name="q" placeholder="Describe what you want to find…"
         value="{{ query or '' }}" autofocus />
  <label>
    Caption ⇄ Visual
    <input type="range" name="image_weight" min="0" max="1" step="0.05"
           value="{{ default_image_weight }}" />
  </label>
  <button type="submit">Search</button>
</form>
```

- [ ] **Step 4: Restructure the form**

In `src/pixsage/web/templates/index.html`, **replace** the form above with:

```jinja
<form id="search-form" method="get" action="/">
  <div class="search-row">
    <input type="search" name="q" placeholder="Describe what you want to find…"
           value="{{ query or '' }}" autofocus />
    <button type="submit">Search</button>
  </div>
  <div class="weight">
    <label for="image-weight">Caption ⇄ Visual</label>
    <input id="image-weight" type="range" name="image_weight"
           min="0" max="1" step="0.05" value="{{ default_image_weight }}" />
  </div>
</form>
```

- [ ] **Step 5: Run the new test + the search suite**

Run: `pytest tests/test_web_catalogs.py::test_home_search_slider_lives_in_weight_row tests/test_web_search.py -v`
Expected: all pass. `test_web_search.py` keys off the `name="q"` and `name="image_weight"` attributes, both preserved.

- [ ] **Step 6: Commit**

```bash
git add src/pixsage/web/templates/index.html tests/test_web_catalogs.py
git commit -m "refactor(web): split search form into search-row + weight row

Slider moves below the input on its own row; input + button stay
on the top row. Selectors and name attributes unchanged."
```

---

## Task 5: Update palette + header + page background in `style.css`

**Files:**
- Modify: `src/pixsage/web/static/style.css`

No test — this is presentation polish. Visual verification happens in Task 8.

- [ ] **Step 1: Read the current `style.css`**

Run: `cat src/pixsage/web/static/style.css`

You'll be touching two regions, identified by content (not line number):

- Top of file: the `body`, `header`, `header h1`, and `main` rules.
- Around the middle: the `header a { color: #58a6ff; ... }` rule and the `header nav { float: right; ... }` rule.

- [ ] **Step 2: Replace the body/header/h1/main rules at the top**

Find these four contiguous lines at the very top of `style.css`:

```css
body { font-family: system-ui, sans-serif; margin: 0; background: #111; color: #eee; }
header { padding: 1rem 1.5rem; border-bottom: 1px solid #333; }
header h1 { margin: 0; font-size: 1.5rem; }
main { padding: 1.5rem; }
```

Replace them with:

```css
body { font-family: system-ui, -apple-system, sans-serif; margin: 0; background: #0d0d0d; color: #eee; }
header { display: flex; align-items: baseline; justify-content: space-between; padding: 0.75rem 1.5rem; border-bottom: 1px solid #1f1f1f; }
header h1 { margin: 0; font-size: 1.25rem; font-weight: 600; }
header h1 a { color: inherit; text-decoration: none; }
main { padding: 1.5rem; max-width: 1400px; margin: 0 auto; box-sizing: border-box; }
```

- [ ] **Step 3: Replace the header nav rules**

Find these two lines (currently around the middle of the file, just before the `/* Explore (cluster grid) */` comment):

```css
header a { color: #58a6ff; text-decoration: none; }
header nav { float: right; margin-top: -2rem; font-size: 0.9rem; }
```

Replace them with:

```css
header nav { margin: 0; font-size: 0.9rem; }
header nav a { color: #58a6ff; text-decoration: none; margin-left: 1rem; }
```

(The flex `header` rule handles right-alignment without `float`.)

- [ ] **Step 4: Verify no test broke**

Run: `pytest tests/test_web_app.py tests/test_web_catalogs.py tests/test_web_search.py -v`
Expected: all pass (CSS changes don't affect HTML assertions).

- [ ] **Step 5: Commit**

```bash
git add src/pixsage/web/static/style.css
git commit -m "style(web): deeper bg + flex header for the restyle

Body bg #111 -> #0d0d0d; header uses flex instead of float-nav;
h1 tightened to 1.25rem/600."
```

---

## Task 6: Style the catalogs-strip + search form

**Files:**
- Modify: `src/pixsage/web/static/style.css`

- [ ] **Step 1: Replace the existing `#search-form` rules**

Find the current rules in `style.css`:

```css
#search-form { display: flex; gap: 1rem; align-items: center; margin-bottom: 1.5rem; }
#search-form input[type=search] { flex: 1; padding: 0.5rem; font-size: 1rem; background: #222; color: #eee; border: 1px solid #444; }
#search-form button { padding: 0.5rem 1rem; background: #2a8; color: white; border: 0; cursor: pointer; }
```

Replace them with:

```css
/* ── Search form ────────────────────────────────────────────────── */
#search-form { display: flex; flex-direction: column; gap: 0.6rem; margin-bottom: 1.5rem; }
#search-form .search-row { display: flex; gap: 0.6rem; align-items: stretch; }
#search-form input[type=search] {
  flex: 1; padding: 0.85rem 1.1rem; font-size: 1.05rem;
  background: #202020; color: #eee; border: 1px solid #2a2a2a;
  border-radius: 999px; outline: none;
}
#search-form input[type=search]::placeholder { color: #777; }
#search-form input[type=search]:focus { border-color: #58a6ff; }
#search-form button[type=submit] {
  padding: 0 1.4rem; background: #2a8; color: #0d0d0d;
  border: 0; border-radius: 999px; cursor: pointer;
  font-weight: 600; font-size: 0.95rem; letter-spacing: 0.2px;
}
#search-form button[type=submit]:hover { background: #36b993; }
#search-form .weight { display: flex; align-items: center; gap: 0.75rem; padding: 0 0.6rem; font-size: 0.85rem; color: #999; }
#search-form .weight label { color: #999; }
#search-form .weight input[type=range] {
  flex: 1; max-width: 260px;
  accent-color: #58a6ff;
}
```

- [ ] **Step 2: Add the catalogs-strip rules**

Add this block to `style.css` after the search-form block:

```css
/* ── Collapsed catalogs strip (above search) ───────────────────── */
.catalogs-strip {
  display: flex; align-items: center; gap: 0.5rem;
  padding: 0.5rem 0.75rem; margin-bottom: 1rem;
  font-size: 0.85rem; color: #999;
  background: #141414; border: 1px solid #1f1f1f; border-radius: 6px;
}
.catalogs-strip .cs-count { color: #ddd; font-weight: 500; }
.catalogs-strip .cs-sep { color: #555; }
.catalogs-strip .cs-path {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.8rem; color: #999;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  min-width: 0; flex: 1;
}
.catalogs-strip .cs-manage {
  background: transparent; border: 0; color: #58a6ff;
  cursor: pointer; font: inherit; padding: 0; margin-left: auto;
}
.catalogs-strip .cs-manage:hover { text-decoration: underline; }
```

- [ ] **Step 3: Run the suite (sanity)**

Run: `pytest tests/test_web_catalogs.py tests/test_web_search.py tests/test_web_results_grid.py -v`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/pixsage/web/static/style.css
git commit -m "style(web): catalogs strip + pill search form

Strip is a muted one-liner. Search input becomes a pill; slider
moves to its own muted row below."
```

---

## Task 7: Style the catalogs modal + list rows + folder picker

**Files:**
- Modify: `src/pixsage/web/static/style.css`

- [ ] **Step 1: Append the modal frame + list rules**

Add to `style.css` (at the end, after the cluster rules — these are scoped under their own selectors so order doesn't matter):

```css
/* ── Catalogs modal ────────────────────────────────────────────── */
dialog.catalogs-modal {
  width: min(560px, calc(100vw - 2rem));
  padding: 0; border: 1px solid #2a2a2a; border-radius: 8px;
  background: #181818; color: #eee;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6);
}
dialog.catalogs-modal::backdrop { background: rgba(0, 0, 0, 0.55); }
.catalogs-modal-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.85rem 1.1rem; border-bottom: 1px solid #2a2a2a;
}
.catalogs-modal-header h2 { margin: 0; font-size: 1rem; font-weight: 600; color: #eee; }
.modal-close {
  background: transparent; border: 0; color: #777; cursor: pointer;
  font-size: 1.4rem; line-height: 1; padding: 0 0.25rem;
}
.modal-close:hover { color: #eee; }
.catalogs-modal-body { padding: 0.9rem 1.1rem 1.1rem; display: flex; flex-direction: column; gap: 0.75rem; }

/* List of catalogs inside the modal */
.catalog-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.25rem; }
.catalog-row {
  display: grid;
  grid-template-columns: 10px minmax(80px, 130px) 1fr auto auto auto;
  gap: 0.6rem; align-items: center;
  padding: 0.4rem 0.5rem; border-radius: 4px;
}
.catalog-row:hover { background: #202020; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.status-available { background: #2ea043; }
.status-offline { background: #555; }
.rename-form { display: inline; }
.rename-form input {
  width: 100%; box-sizing: border-box;
  background: #202020; color: #eee;
  border: 1px solid #2a2a2a; border-radius: 3px;
  padding: 0.25rem 0.4rem; font: inherit; font-size: 0.85rem;
}
.rename-form input:focus { border-color: #58a6ff; outline: none; }
.catalog-row .path {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.78rem; color: #888;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0;
}
.catalog-row input[type=checkbox] { accent-color: #2ea043; cursor: pointer; }
.catalog-row input[type=checkbox]:disabled { cursor: not-allowed; opacity: 0.5; }
.offline-tag {
  font-size: 0.7rem; color: #888;
  background: #222; border: 1px solid #2a2a2a;
  padding: 0.05rem 0.4rem; border-radius: 3px;
  text-transform: uppercase; letter-spacing: 0.5px;
}
.remove-btn {
  background: transparent; border: 0; color: #666;
  cursor: pointer; font-size: 1.05rem; line-height: 1; padding: 0 0.25rem;
}
.remove-btn:hover { color: #f08080; }

.catalog-notice {
  margin: 0; padding: 0.5rem 0.65rem;
  background: #1f1a10; border: 1px solid #3a2f15; color: #d4b96b;
  border-radius: 4px; font-size: 0.85rem;
}
.empty-state {
  margin: 0; padding: 0.75rem 0;
  color: #999; font-size: 0.9rem; line-height: 1.4;
}
.empty-state code { background: #202020; padding: 0.05rem 0.3rem; border-radius: 3px; color: #ccc; font-size: 0.85em; }

/* Modal footer actions */
.catalogs-actions {
  display: flex; gap: 0.5rem; padding-top: 0.6rem;
  border-top: 1px solid #2a2a2a;
}
.catalogs-actions button,
.catalogs-actions form button {
  background: #202020; color: #ddd;
  border: 1px solid #2a2a2a; border-radius: 4px;
  padding: 0.45rem 0.85rem; cursor: pointer;
  font: inherit; font-size: 0.85rem;
}
.catalogs-actions button:hover,
.catalogs-actions form button:hover { background: #2a2a2a; color: #eee; }
```

- [ ] **Step 2: Append the folder-picker (`#catalog-browser`) rules**

Add to `style.css` (after the modal block above):

```css
/* ── Folder-picker dialog (nested inside the catalogs modal) ──── */
dialog#catalog-browser {
  width: min(480px, calc(100vw - 2rem));
  padding: 0; border: 1px solid #2a2a2a; border-radius: 8px;
  background: #181818; color: #eee;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.7);
}
dialog#catalog-browser::backdrop { background: rgba(0, 0, 0, 0.55); }
dialog#catalog-browser h3 {
  margin: 0; padding: 0.85rem 1.1rem;
  border-bottom: 1px solid #2a2a2a;
  font-size: 0.95rem; font-weight: 600;
}
dialog#catalog-browser .cb-current {
  margin: 0; padding: 0.55rem 1.1rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.78rem; color: #888;
  border-bottom: 1px solid #1f1f1f;
  word-break: break-all;
}
dialog#catalog-browser .cb-list {
  list-style: none; margin: 0; padding: 0.4rem 0;
  max-height: 320px; overflow-y: auto;
}
dialog#catalog-browser .cb-list li { display: block; }
dialog#catalog-browser .cb-list a {
  display: block; padding: 0.35rem 1.1rem;
  color: #ddd; text-decoration: none; font-size: 0.88rem;
}
dialog#catalog-browser .cb-list a:hover { background: #202020; color: #fff; }
dialog#catalog-browser form {
  display: flex; gap: 0.5rem;
  padding: 0.7rem 1.1rem; border-top: 1px solid #2a2a2a;
}
dialog#catalog-browser form button[type=submit] {
  background: #2a8; color: #0d0d0d; border: 0; border-radius: 4px;
  padding: 0.45rem 0.85rem; cursor: pointer; font: inherit; font-size: 0.85rem; font-weight: 600;
}
dialog#catalog-browser form button[type=submit]:disabled { background: #1f3a30; color: #5a7a70; cursor: not-allowed; }
dialog#catalog-browser form button[type=button] {
  background: #202020; color: #ddd; border: 1px solid #2a2a2a;
  border-radius: 4px; padding: 0.45rem 0.85rem; cursor: pointer; font: inherit; font-size: 0.85rem;
}
dialog#catalog-browser form button[type=button]:hover { background: #2a2a2a; }
```

- [ ] **Step 3: Run the suite (sanity)**

Run: `pytest tests/test_web_catalogs.py tests/test_web_search.py tests/test_web_results_grid.py tests/test_web_app.py tests/test_web_app_multi.py -v`
Expected: all pass — CSS-only changes.

- [ ] **Step 4: Commit**

```bash
git add src/pixsage/web/static/style.css
git commit -m "style(web): catalogs modal + folder-picker dialog

Both dialogs share the same dark frame; modal list rows use a
6-col grid (dot / label / path / toggle / offline-tag / remove)."
```

---

## Task 8: Manual verification on the live runtime

**Files:** none — this is a visual smoke test.

- [ ] **Step 1: Re-stage the runtime**

Run:
```bash
rsync -a --delete src/pixsage/ "$HOME/Library/Application Support/pixsage/runtime/site-packages/pixsage/"
find "$HOME/Library/Application Support/pixsage/runtime/site-packages/pixsage/" -name __pycache__ -type d -exec rm -rf {} +
```

- [ ] **Step 2: Launch the app**

Open `~/Applications/Pixsage Search.command` (or run the launcher manually). Wait for the loading screen to flip to the search page.

- [ ] **Step 3: Walk through the verification checklist**

Confirm each (mark on the checkbox above only once all pass):

1. Page is dark (deeper than before), header is one tight line.
2. Search input is a pill, centred in the column, with the green **Search** button to its right.
3. Caption ⇄ Visual slider sits on its own muted row below the search input.
4. Above the search: a single muted line — `1 catalog · /Volumes/T7/2026/.photoindex   Manage ▸`.
5. Click **Manage ▸** → a centred dark modal opens with a `Catalogs (N)` header, a close `×`, the catalog row(s), and **Add catalog…** / **Refresh availability** in the footer.
6. ESC closes the modal. Re-open it. Click outside the modal box (on the backdrop) → it closes.
7. Re-open. Rename a catalog (blur submits, page reloads, modal closed — that's expected).
8. Re-open. Click **Add catalog…** → the folder-picker dialog opens *on top of* the management modal. Navigate, **Cancel** closes the picker but leaves the management modal open.
9. Submit a search. Result cards tile (grid), not stacked full-width.
10. Move the slider, submit again. The new `image_weight` is honoured.

- [ ] **Step 4: Final commit (only if any small polish tweaks emerged in step 3)**

If steps 1–10 all pass cleanly, no commit needed — the work is done. If you tweaked CSS during verification, commit that as `style(web): polish from manual verification`.

---

## Out of scope (do NOT do)

- Restyling `cluster.html`, `explore.html`, `photo.html`.
- HTMX partial-replacement of results.
- Adding a light/dark theme switch.
- Adding new catalog-management features (the modal exposes exactly the same actions the panel did).
- Refactoring `routes.py` or `app.py`.
