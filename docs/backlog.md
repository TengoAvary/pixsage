# Backlog

Feature ideas captured but not yet scheduled. Not blocking photographer handoff. Order is rough — pick by signal, not by position.

## Search

### Boolean / multi-phrase search (AND, OR)

**Idea:** let the user combine phrases — `penguin AND iceberg`, `seal OR walrus`, possibly grouped. Each phrase runs as an independent similarity search; results are combined in score space.

**Why it'd be useful:** single-phrase queries are blunt. The photographer often wants "X plus Y" (two subjects in one frame) or "X or Y" (browse a category). Today you'd run two searches and reconcile by eye.

**Sketch:**
- **Parser**: split on `AND` / `OR` (case-insensitive, whitespace-delimited). MVP: no parentheses, no precedence — flat list with one operator type per query. Reject mixed AND/OR until grouping syntax is decided.
- **Execution**: run `SearchService.search()` per phrase in parallel (asyncio or threadpool — encoder calls are GIL-releasing torch ops). Each returns top-k with scores.
- **Combine**:
  - OR → union by sha; score = max across components. Re-sort, take top-k.
  - AND → intersection; score = mean of component scores (or min, to be conservative). Re-sort, take top-k.
- **Top-k inflation**: for AND, top-k of each component is rarely enough — the intersection can be empty. Fetch top-N where N = k × 4 or so for each component, then intersect. Cheap on a 1.5k-photo corpus.
- **UX**: same input field, parser handles operators. Surface them in the placeholder ("e.g. penguin AND iceberg"). No new UI element required for MVP. If the parser rejects (mixed ops), render a clear inline message instead of results.

**Open questions:**
- Quoted multi-word phrases (`"king penguin" AND iceberg`)? Probably yes — Florence-2 captions are sentence-shaped, so multi-word phrases matter.
- NOT operator? Skip for MVP; usefulness uncertain. Easy to add later as a per-phrase score subtraction.
- Slider behavior when multiple phrases: apply the same `image_weight` to all sub-searches. Per-phrase weight would be overkill.

**Effort:** ~half day if AND/OR stay flat. Most of the work is parser + intersect/union; the search service already exposes the right primitives.

### Saved searches panel

**Idea:** pin frequently-used queries to a side panel. Click to drop the query (and slider value) back into the search box. Optionally one-click runs it.

**Why it'd be useful:** the photographer has recurring lookups ("emperor penguin chicks", "research station exteriors", etc.) and rebuilds them every session. Pin once, reuse forever.

**Sketch:**
- **Storage**: new `saved_searches` table in the catalog SQLite (lives in `.photoindex/`, travels with the drive — so saves are per-corpus, which is the right scope). Columns: `id`, `query`, `image_weight`, `label` (optional human name), `created_at`.
- **API**:
  - `GET /saved` — list, returns JSON or HTML partial.
  - `POST /saved` — save current `(q, image_weight, label?)`.
  - `DELETE /saved/{id}` — remove.
- **UX**:
  - "★" button next to the Search button. Click → POST `/saved` with current query state. Optional prompt for a label.
  - Side panel (or collapsible) listing saved searches. Click a row → fills the input + slider, optionally auto-submits.
  - Render the panel on `index.html`; empty state is fine ("No saved searches yet").
- **Persistence on the drive matters**: when the photographer opens the drive on a different machine, the saved searches come with the catalog. The path-translation work from Plan 1 ensures the catalog itself is portable — saved searches sit on top of that for free.

**Open questions:**
- Should saved searches survive `pixsage embed --rewrite` etc.? Yes — they're independent of vectors/tags.
- Edit a saved search (change label, tweak query)? Defer. Delete + re-save is enough for MVP.
- Order: most-recently-used vs alphabetical? MRU on access is friendliest; needs an `last_used_at` column. Skip for MVP — sort by `created_at DESC`.

**Effort:** ~half day. New table + 3 routes + a small panel. Most of the time is the UX polish (panel layout, save flow).

## Notes

Both items are additive — neither touches the search/embed/tag pipelines. They can ship independently.

If both land, the natural follow-up is **saving a boolean query as a saved search** (e.g. "penguin AND iceberg" pinned with one click). The data model supports it as long as `query` is a single string field.
