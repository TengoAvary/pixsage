# Photo Corpus Pipeline — Spec

Status: draft, 2026-05-09
Context: app being developed for a photographer; designed so the tagging tier is immediately useful in Lightroom while the richer tier feeds a portable visualisation/search app.

## Goals

1. **Tier 1 — immediate utility.** Auto-tag a photographer's full corpus with keywords that appear natively in Lightroom, without disrupting their existing catalog or workflow.
2. **Tier 2 — richer structure.** Produce per-photo embeddings, captions, geo estimates, and a similarity graph supporting clustering and semantic search.
3. **Portable, drive-resident catalog.** All derived metadata travels with the photos. Plug the drive into another machine, launch a web app, get full functionality. No re-processing required.
4. **Robust to reorganisation.** Files identified by content hash, not path. Renames, moves, copies, and re-encodes do not break the catalog.

## Architecture overview

```
┌─────────────────────────────────────────────────────────────┐
│  Photographer's drive                                        │
│                                                              │
│  /photos/...           ← original raws + JPEGs               │
│    DSC_0001.ARW                                              │
│    DSC_0001.xmp        ← Lightroom-native sidecar (Tier 1)   │
│                                                              │
│  /.photoindex/         ← portable catalog (Tier 2)           │
│    catalog.db          ← SQLite: hashes, paths, tags, EXIF   │
│    embeddings.parquet  ← image + text embeddings             │
│    graph.parquet       ← edge list of similarity graph       │
│    clusters.json       ← cluster assignments + labels        │
│    config.json         ← model versions, pipeline params     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌─────────────────────────────┐
              │  Web app (local server)      │
              │  - UMAP scatter view         │
              │  - cluster browser           │
              │  - semantic search           │
              │  - keyword filter            │
              │  - map view (geo estimates)  │
              └─────────────────────────────┘
```

Two pipelines write into the same catalog. Web app reads from it. Lightroom reads only the XMP sidecars and stays oblivious to `.photoindex/`.

## Tier 1: Lightroom-native tagging

Goal: get usable keywords into Lightroom with zero workflow disruption.

**Models (candidates, decide via small benchmark):**
- **Florence-2** (770M, MIT) — captioning, object detection, dense captioning, OCR. Default choice for raw throughput; ~30–100 img/sec on a 4090.
- **RAM++** — purpose-built tagger, returns thousands of canonical tags per image. Good as a complement to Florence-2 since the tag vocabularies are different.

**Output to XMP:**
- Flat keywords → `dc:subject` (Dublin Core, Lightroom reads as standard keywords)
- Hierarchical tags → `lr:hierarchicalSubject` (renders as nested tree in Lightroom keyword panel)
- Generated caption → `dc:description` (visible in Lightroom metadata panel)
- A namespace tag like `auto-tagged` so the photographer can filter for/against AI tags

For raws (.ARW, .CR3, etc.) write a sidecar `.xmp` next to the file. For JPEG/TIFF/HEIC/DNG, embed directly in the file. Use `exiftool` as the canonical writer.

**Lightroom sync:**
- Best UX: photographer enables Catalog Settings → Metadata → "Automatically write changes into XMP" once. Catalog stays in sync.
- Otherwise: select photos → Metadata → Read Metadata from File.

**Tag vocabulary curation:**
- Don't dump raw model output. Maintain a confidence threshold and an exclusion list per project.
- Hierarchical structure should be consistent: `Scene|Mountain|Glacier`, `Wildlife|Bird|Penguin`, `People|Group`, `Time|Golden Hour`.
- Optionally let photographer review/edit the vocabulary on a sample before committing to the full corpus.

## Tier 2: Full processing

Runs in addition to Tier 1. Produces the data backing the web app.

**Per-photo outputs:**

| Field             | Source                              | Notes                              |
|-------------------|-------------------------------------|------------------------------------|
| sha256            | file bytes                          | primary key                        |
| phash             | resized greyscale image             | secondary key, robust to re-encode |
| exif              | embedded EXIF                       | timestamp, camera, lens, settings  |
| gps               | EXIF if present                     | nullable                           |
| caption           | VLM (Moondream2 or Qwen2.5-VL 7B)   | one rich sentence                  |
| tags              | Florence-2 / RAM++                  | flat + hierarchical                |
| image_embedding   | SigLIP2 or OpenCLIP                 | ~768–1024 dim                      |
| geo_embedding     | StreetCLIP or GeoCLIP               | for geo-aware similarity           |
| geo_estimate      | GeoCLIP inference                   | predicted lat/lon + confidence     |
| text_embedding    | BGE or similar, on caption          | for semantic search                |

**Graph + clustering:**
- Build mutual k-NN graph (k=15–30) over a fused embedding (concat or weighted sum of image, geo, and text embeddings — tune weights empirically).
- UMAP → HDBSCAN for clustering. HDBSCAN handles varying density and noise rejection without pre-specified cluster count.
- Persist UMAP 2D coordinates per photo for the visualisation.
- Label each cluster automatically: top tags by TF-IDF within cluster, plus most common GeoCLIP region prediction.

**Hardware split:**
- Run Tier 2 on the 4090. Roughly: thousands of photos in tens of minutes for embeddings; hours if including a 7B VLM caption pass. Florence-2 tagging is essentially free on top.
- MacBook can run Tier 1 alone for incremental updates on small batches.

## File identification strategy

The robustness requirement. Three-layer fallback:

1. **SHA-256 of file bytes** — primary key. Survives renames, moves, copies. Breaks if file is re-saved (edited, re-encoded, metadata-stripped).
2. **Perceptual hash (pHash)** — secondary index. Survives re-encodes, format conversion, mild edits. Won't match heavy crops/edits but catches the common cases (JPEG re-export, rotation).
3. **EXIF triple** (camera serial + capture timestamp + original filename) — tertiary fallback. Useful when both hashes change (e.g. heavy retouching) but EXIF preserved.

**Lookup flow on a new machine:**
1. Walk drive → enumerate image files
2. For each: compute sha256, look up in catalog → match? Update path, done.
3. If no match: compute phash, lookup → match? Mark as "edited derivative of X", update path, done.
4. If no match: try EXIF triple.
5. If no match: it's a new photo, queue for ingestion.

**Catalog stores the *current* path as a hint, but never relies on it.** Every web app session re-scans on launch (cheap if catalog is recent — only check files whose mtime is newer than last scan).

## Catalog format

**SQLite** for the relational metadata + index (single file, queryable, portable, supported everywhere including a JS client via sql.js or DuckDB-WASM).

**Parquet** for the embedding matrix and graph edge list (columnar, efficient for numeric data, readable from Python, JS, and DuckDB).

Schema sketch:

```sql
CREATE TABLE photos (
  sha256 TEXT PRIMARY KEY,
  phash BLOB,
  current_path TEXT,
  filename TEXT,
  filesize INTEGER,
  capture_time TEXT,
  camera_make TEXT,
  camera_model TEXT,
  camera_serial TEXT,
  lens TEXT,
  gps_lat REAL,
  gps_lon REAL,
  caption TEXT,
  cluster_id INTEGER,
  umap_x REAL,
  umap_y REAL,
  geo_estimate_lat REAL,
  geo_estimate_lon REAL,
  geo_estimate_confidence REAL,
  added_at TEXT,
  last_seen_at TEXT
);

CREATE TABLE tags (
  sha256 TEXT,
  tag TEXT,
  source TEXT,        -- 'florence2', 'ram++', 'manual'
  confidence REAL,
  hierarchy TEXT,     -- nullable, e.g. 'Wildlife|Bird|Penguin'
  PRIMARY KEY (sha256, tag, source)
);

CREATE TABLE clusters (
  cluster_id INTEGER PRIMARY KEY,
  label TEXT,
  top_tags TEXT,
  representative_sha256 TEXT,
  size INTEGER,
  geo_centroid_lat REAL,
  geo_centroid_lon REAL
);

CREATE INDEX idx_phash ON photos(phash);
CREATE INDEX idx_cluster ON photos(cluster_id);
CREATE INDEX idx_capture_time ON photos(capture_time);
```

Embeddings stored separately in `embeddings.parquet` keyed by sha256 — keeps the SQLite small and lets numeric tools work on the matrix directly.

## Web app

**Stack (proposed):**
- Backend: FastAPI (Python) — needed because some operations (re-running search against text query embedding, on-the-fly hashing of new files) want server-side compute.
- Frontend: React + deck.gl for the UMAP scatter (handles 100k+ points), Leaflet for the geo map view, plain components for the rest.
- Catalog read via SQLAlchemy + pyarrow.
- Launched as a local-only server: `photoapp serve /path/to/drive` → opens browser at `localhost:PORT`.

**Views:**
- **Map** — photos plotted at GPS (real or estimated), clusters coloured.
- **UMAP scatter** — every photo a point in 2D embedding space, coloured by cluster. Click to inspect, lasso-select to bulk-tag or export.
- **Cluster browser** — list of clusters with auto-generated labels, sample photos, drill-in.
- **Search** — text query → embed via same text encoder → nearest neighbours by cosine.
- **Keyword filter** — facet over the `tags` table.
- **Photo detail** — full metadata, similar photos, "where on the map", "where in UMAP".

**Stateless against drive moves:** on launch, web app verifies catalog against current drive contents using the three-layer hash lookup. Updates `current_path` for moved files. Marks missing files. Queues new files for ingestion if the user opts in.

## Implementation phases

**Phase 1 — Tier 1 MVP (1 week of evening work):**
- exiftool wrapper, Florence-2 inference, XMP writer.
- CLI: `photoapp tag /path/to/photos`.
- Test on a small batch, verify Lightroom picks up keywords cleanly.

**Phase 2 — Catalog + identification (1 week):**
- SQLite schema, file walker, sha256 + phash computation.
- Re-identification logic.
- CLI: `photoapp index /path/to/photos`.

**Phase 3 — Tier 2 embeddings (1–2 weeks):**
- SigLIP2 + GeoCLIP + caption pipelines.
- Parquet writer.
- UMAP + HDBSCAN.

**Phase 4 — Web app (2 weeks):**
- FastAPI backend, React frontend.
- Map, UMAP, search, cluster browser.

**Phase 5 — Polish:**
- Incremental re-runs (don't reprocess unchanged files).
- Batch tag editing UI that writes back to XMP.
- Cluster naming UI for photographer-curated labels.
- Export: cluster as Lightroom collection, smart album definitions.

## Open questions / decisions

- **Tagging models:** Florence-2 alone, RAM++ alone, or both? Worth a small benchmark on representative photos from the photographer's actual corpus.
- **Captioner choice:** Moondream2 (fast, simpler) vs Qwen2.5-VL 7B (richer, slower). Probably Moondream2 for first cut.
- **Embedder:** SigLIP2 is a stronger general embedder; GeoCLIP gives location signal. Default: run both, fuse.
- **Run on raws or derivatives?** Raws are slow to decode and large. Generating reduced JPEGs first (1024px long edge) speeds tagging considerably; quality loss for tagging purposes is negligible. Decision: pre-generate JPEG cache, process from cache, but link results back to raw via path.
- **Where does `.photoindex/` actually live?** On the drive itself for portability. But if the photographer uses cloud sync (Dropbox/iCloud over the photo dir), consider a sibling directory the user explicitly opts in to syncing.
- **Multi-user / multi-machine writes?** Out of scope for v1. Single writer, multi-reader.
- **Privacy:** all processing local; no cloud calls. Make this an explicit selling point in the app description.

## Stretch ideas

- **Time-aware clustering.** Add capture timestamp as a feature; "events" emerge naturally (a wedding day forms a tight cluster; a multi-week trip forms a band).
- **Anchor-based geolocation.** When some photos in a cluster have real GPS, propagate the location to GPS-less neighbours via the graph.
- **Style fingerprints per photographer.** Train a small classifier on the embedding space against "this photographer vs others" to surface their stylistic signatures.
- **"Find similar photos" Lightroom plugin.** Read embeddings from `.photoindex/`, expose a "find similar" right-click action inside Lightroom. Closes the loop.
