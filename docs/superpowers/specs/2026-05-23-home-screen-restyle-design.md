# Home-screen restyle: search-first layout + catalogs modal

**Date:** 2026-05-23
**Status:** Design

## Background

The current home page (`/`) renders three regions stacked vertically:

1. A `<details class="catalogs-panel">` containing the catalog list, rename inputs, enable/disable toggles, remove buttons, **Add catalog…**, **Refresh availability**, and a nested folder-picker `<dialog id="catalog-browser">`.
2. A `<form id="search-form">` with the search input, a Caption ⇄ Visual range slider, and a Search button.
3. A `<section id="results">` containing the partial `_results.html`.

Two visible problems:

- **`_catalogs.html` ships with no CSS.** Class names like `.catalogs-panel`, `.catalog-list`, `.catalog-row`, `.status-dot`, `.path`, `.rename-form input`, `.catalogs-actions`, `#catalog-browser`, `.cb-list` are referenced in markup but have zero matching rules in `style.css`. Every element falls back to raw browser defaults — white text inputs, bullet lists, OS-gray buttons. (Verified: no commit has ever added these selectors to `style.css`.)
- **The results grid is broken.** `_results.html` emits `<div class="grid">…</div>` but `style.css` only grids `#results`, so result cards stack full-width rather than tiling.

This is the "make it not ugly" work that has been sitting open since 2026-05-17.

## Goals

- Style every currently-unstyled element on the home page so the dark theme reads as intentional, not half-built.
- Reshape the home page to put **search first** and demote catalog management to a single line that opens a modal on demand.
- Fix the results-grid CSS/markup mismatch.
- Keep behaviour, routes, and existing tests intact — this is a presentation-layer change plus one structural swap (`<details>` → `<dialog>`).

## Non-goals

- Cluster pages (`cluster.html`, `explore.html`), which already have working CSS.
- The photo-detail page (`photo.html`).
- HTMX partial-replacement of results.
- Any light-mode / theme-switching mechanism.
- New catalog-management features. The modal must expose exactly the same actions the current `<details>` panel exposes.

## Aesthetic direction

Search-first / gallery (direction "B" in brainstorming). The user's primary verb on this page is **search**; catalog management is occasional configuration. Photos should get the most page real-estate; catalog tooling should be present but quiet.

### Palette

| Token | Value | Use |
| --- | --- | --- |
| `bg` | `#0d0d0d` | Page background (slightly deeper than current `#111` to make photos pop) |
| `surface` | `#181818` | Modals, raised panels |
| `surface-hi` | `#202020` | Inputs, hover states |
| `border` | `#222` | Subtle dividers |
| `border-emph` | `#2a2a2a` | Card edges, modal frame |
| `text` | `#eee` | Primary text |
| `text-muted` | `#999` | Secondary text, meta strip |
| `text-dim` | `#666` | Tertiary, placeholder, offline |
| `link` | `#58a6ff` | Links, "Manage ▸" affordance *(existing)* |
| `accent` | `#2a8` | Search button *(existing)* |
| `ok` | `#2ea043` | Status dot when catalog available *(existing)* |
| `bad` | `#b00020` | Errors (reserved; already in loading.css) |

Values are inlined in `style.css` (no CSS variables for now — the project uses plain CSS, keep it that way).

### Typography

System UI stack, unchanged. Size adjustments:

- `header h1`: `1.25rem`, weight `600` (was `1.5rem`)
- Hero search input: `1.05rem`, `1rem` vertical padding
- Meta strip: `0.85rem`, muted
- Modal title: `1rem`, weight `600`
- Modal body text: `0.9rem`

## Home-page layout

```
┌──────────────────────────────────────────────┐
│  pixsage                                     │  compact header
├──────────────────────────────────────────────┤
│                                              │
│   ┌──────────────────────────────────────┐   │  hero search
│   │ Describe what you want to find…      │   │  (pill, autofocus)
│   └──────────────────────────────────────┘   │
│                                              │
│   1 catalog · /Volumes/T7        Manage ▸    │  catalogs-strip
│   Caption ⇄ ●─────── Visual                  │  weight slider
│                                              │
│   ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓  │  results grid
└──────────────────────────────────────────────┘
```

### `index.html` changes

- The `{% include "_catalogs.html" %}` block continues to render at the top of `<main>`, but `_catalogs.html` is restructured (see below) so what's *shown by default* is the one-line `.catalogs-strip`, not the full panel.
- The search `<form>` is restructured: search input on its own line with the **Search** button sitting *to the right of* the input on the same row (not absolutely positioned inside the pill). The Caption ⇄ Visual slider moves into a sibling `<div class="weight">` row *below* the input row.
- Tests that hit `#search-form`, the search `<input name="q">`, and the slider `<input name="image_weight">` continue to pass — selectors / names unchanged.

### `_results.html` changes

- Delete the wrapping `<div class="grid">…</div>`. Emit the per-hit cards directly.
- `<section id="results">` in `index.html` already has `display: grid`, so the cards tile under the existing rule.
- The `.grid` class is removed from `style.css` (it has no rule today, but ensure no stale reference is left in any template).

## Collapsed catalogs strip

A single line replacing the current full panel on the home page:

```html
<div class="catalogs-strip">
  <span class="cs-count">{{ entries|length }} catalog{{ 's' if entries|length != 1 }}</span>
  <span class="cs-sep">·</span>
  <span class="cs-path">{{ first_available_path or 'none mounted' }}</span>
  <button type="button" class="cs-manage" data-action="open-catalogs">Manage ▸</button>
</div>
```

States:

- **At least one available catalog**: `1 catalog · /Volumes/T7   Manage ▸`. `Manage ▸` opens the modal.
- **Entries exist but none available**: `1 catalog · offline   Manage ▸`. Manage still opens.
- **No entries at all**: `No catalogs yet   Add one ▸`. The link opens the management modal (does **not** auto-open the folder picker — the user clicks **Add catalog…** inside the modal). One affordance, one click, predictable.

The strip is always rendered when the registry is present (the existing `{% if registry %}` guard in `index.html` keeps controlling whether catalog UI shows at all).

## Catalogs modal

`_catalogs.html` becomes a `<dialog>` instead of a `<details>`. The inner markup — list of catalogs, rename forms, toggle forms, remove forms, **Add catalog…**, **Refresh availability**, the nested folder-picker `<dialog id="catalog-browser">`, the existing inline `<script>` — is preserved as-is so all form actions (`/catalogs/{id}/rename`, `/toggle`, `/remove`, `/refresh`, `/add-scan`, `/catalogs/browse`) keep working.

```html
<dialog class="catalogs-modal" id="catalogs-modal">
  <header>
    <h2>Catalogs</h2>
    <button type="button" class="modal-close" aria-label="Close"
            onclick="this.closest('dialog').close()">×</button>
  </header>
  <div class="catalogs-modal-body">
    {# existing notice, catalog-list / empty-state, catalogs-actions, nested folder-picker dialog, inline script #}
  </div>
</dialog>
```

Behaviour:

- The collapsed strip's `Manage ▸` button calls `document.getElementById('catalogs-modal').showModal()`. One small inline handler in `_catalogs.html` (next to the existing script) covers the open path and a backdrop-click-to-close.
- ESC closes the modal (native `<dialog>` behaviour).
- The nested folder-picker `<dialog id="catalog-browser">` continues to live inside the modal body. Native `<dialog>` supports stacking — opening it on top of the management modal works without changes. Closing the picker returns focus to the management modal.
- Forms inside the modal continue to POST and rely on the server's existing redirect to `/`. The page reload naturally returns the modal to its closed default state, which matches the "I'm done managing" intent.

## CSS additions to `style.css`

All additions are scoped under specific selectors — no global resets, no changes to existing rules for cluster/photo pages.

| Selector | Notes |
| --- | --- |
| `body` | Update `background` to `#0d0d0d`. Existing rule. |
| `header`, `header h1`, `header a`, `header nav` | Tighten padding, smaller h1, remove the float-hack on nav (use flex) |
| `.catalogs-strip` | Flex row, gap, muted text, push `Manage ▸` to the right |
| `.cs-count`, `.cs-sep`, `.cs-path` | Muted text colours; `.cs-path` clips with `text-overflow: ellipsis` |
| `.cs-manage` | Link-styled button (`#58a6ff`, no background, no border) |
| `#search-form` | Vertical flex, gap. The input row sits above the weight row |
| `#search-form input[type=search]` | Pill: large padding, `border-radius: 999px`, `surface-hi` background |
| `#search-form button[type=submit]` | Accent (`#2a8`), rounded, sits on the right of the pill row |
| `#search-form .weight` | New row: small muted label, range input with accent thumb |
| `#search-form input[type=range]` | Custom track/thumb using `surface-hi` and `accent` |
| `dialog.catalogs-modal` | 520px wide, centred, `surface`, rounded, soft border. Reset default `<dialog>` padding |
| `dialog.catalogs-modal::backdrop` | `rgba(0,0,0,0.55)` |
| `.catalogs-modal header` | Flex: title left, close × right; bottom border |
| `.catalogs-modal h2` | `1rem`, weight 600 |
| `.modal-close` | Borderless icon button, dim → text on hover |
| `.catalogs-modal-body` | Padding, gap |
| `.catalog-list` | Plain list (no bullets), `gap` between rows |
| `.catalog-row` | Grid: `8px 1fr auto auto auto` (dot, label+path, toggle, offline-tag, remove). Hover bg `surface-hi` |
| `.status-dot` | 8px circle, inline-block |
| `.status-available` | `background: #2ea043` |
| `.status-offline` | `background: #555` |
| `.rename-form input` | `surface-hi` bg, subtle border, full width inside its cell |
| `.path` | Monospace, muted, ellipsis |
| `.offline-tag` | Small muted pill |
| `.remove-btn` | Borderless × button, dim → red-tint on hover |
| `.catalogs-actions` | Flex row, gap, sits below the list inside the modal body |
| `.catalogs-actions button` | Secondary button: outlined, `surface` bg, hover `surface-hi` |
| `.empty-state`, `.catalog-notice` | Muted, small |
| `dialog#catalog-browser` | Same modal frame as `.catalogs-modal`, slightly narrower |
| `dialog#catalog-browser h3` | Match the modal-header style |
| `.cb-current` | Monospace muted, breaks on path |
| `.cb-list` | Plain list, scrollable max-height |
| `.cb-list li` | Hover bg `surface-hi`, pointer cursor |
| `.cb-list a` | Full-row clickable, no underline, `text` colour |

Existing rules to leave alone: `.card`, `.card img`, `.card .meta`, `.photo-detail*`, `.tags*`, `.more-like-this`, all `.cluster-*`, all `.loading-*`.

## Templates touched

| File | Change |
| --- | --- |
| `src/pixsage/web/templates/index.html` | Restructure search form (slider moves to its own row); no other change |
| `src/pixsage/web/templates/_catalogs.html` | Add `.catalogs-strip` block at the top. Replace outer `<details>` with `<dialog class="catalogs-modal" id="catalogs-modal">` containing the existing list / actions / nested picker / script. Add a small inline JS handler on the strip's `Manage ▸` button to call `showModal()` and a backdrop-click-close |
| `src/pixsage/web/templates/_results.html` | Drop wrapping `<div class="grid">` |
| `src/pixsage/web/static/style.css` | Add the rules listed above. Update `body` background. Remove any reference to `.grid` |

No Python changes. Route handlers and Jinja context are unchanged.

## Server-side template context

The collapsed strip needs to know "the first available catalog's path" (or detect that none is available). Two acceptable implementations:

1. Compute it inside the Jinja template from `registry.entries() | selectattr('available') | list | first`.
2. Pass `first_available_entry` from the route handler.

Pick the Jinja-only approach to avoid touching `routes.py`. Loops are tiny (catalogs are usually 1–5).

## Testing

The existing web test suite covers:

- Catalog row markup (rename form, toggle form, remove button) — passes if class names and form actions are preserved.
- `<dialog id="catalog-browser">` presence and the `/catalogs/browse` endpoint — passes since the picker is preserved.
- Search form submission and result rendering — passes since `id`s and `name`s are preserved.

New test:

- `tests/web/test_results_grid.py` (or appended to an existing results test): assert the `_results.html` partial does **not** emit `class="grid"` in its rendered output for a non-empty hit list. Locks the grid-bug fix.

Manual verification on the live runtime after deployment:

1. Restart `pixsage serve`, load `/`. Page is dark, search is the hero, the catalogs strip is one line.
2. Click **Manage ▸** → modal opens centred with backdrop. ESC and backdrop click close it.
3. Inside modal: rename a catalog (blur submits, redirect back, modal closed). Toggle enabled. Remove a catalog. **Refresh availability** works.
4. Inside modal: **Add catalog…** opens the folder picker on top. Navigate, pick a folder, **Scan**. Returns to `/`.
5. Submit a search. Result cards tile in the grid (not stacked full-width).
6. Move the Caption ⇄ Visual slider; submit. The `image_weight` value is honoured (existing test path).

## Risks and mitigations

- **`<dialog>` browser support.** Native `<dialog>` is supported in all current Safari / Chrome / Firefox. Pixsage launches a local browser via the launcher; this is the user's own browser so no IE/legacy concern. No polyfill needed.
- **The nested folder-picker dialog already works inside the current `<details>`.** Moving its parent from `<details>` to `<dialog>` is a structural change but doesn't change the picker's own contract (it's `showModal()`'d by its own button). The native stacking order makes the picker render above its parent modal — verified behaviour, not a hack.
- **Test for `.grid` class absence is narrow.** If a future contributor reintroduces a `.grid` wrapper for some other reason, the test will catch it specifically in `_results.html`. That's the intent.
- **Background colour change (`#111` → `#0d0d0d`)** is a global visual shift. The cluster / photo / loading pages all set their own backgrounds where it matters; spot-check those manually after the change.
