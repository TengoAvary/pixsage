from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from pixsage.registry import DEFAULT_CAPTION_SIGNATURE, DEFAULT_IMAGE_SIGNATURE
from pixsage.web.thumbs import ThumbSize


def register(app: FastAPI, *, experimental_cluster_labelling: bool = False) -> None:
    """Register all routes on `app`.

    `experimental_cluster_labelling` (default off) controls the HITL
    cluster-based location labelling routes (/explore, /cluster/{id},
    /cluster/{id}/label). See the comment above the cluster routes below for
    the why-it-exists / why-it's-disabled context.
    """
    @app.get("/", response_class=HTMLResponse)
    def index(
        request: Request,
        q: str = "",
        image_weight: float | None = None,
        notice: str | None = None,
    ) -> HTMLResponse:
        templates = app.state.templates
        config = app.state.config
        multi = app.state.multi_search
        registry = app.state.registry
        catalogs = app.state.catalogs

        if image_weight is None:
            image_weight = config.search.default_image_weight

        # Build query signatures — for now, the orchestrator's embedder
        # signatures are the defaults. If we ever support multiple query
        # encoders, this changes.
        q_img_sig = DEFAULT_IMAGE_SIGNATURE
        q_cap_sig = DEFAULT_CAPTION_SIGNATURE

        hits: list | None = None
        if q.strip():
            raw_hits = multi.search(
                query=q,
                image_weight=image_weight,
                top_k=config.search.top_k,
                query_image_sig=q_img_sig,
                query_caption_sig=q_cap_sig,
            )
            hits = []
            for h in raw_hits:
                cat = catalogs.get(h.catalog_id)
                if cat is None:
                    continue
                row = cat.get_photo(h.sha256)
                if row is None:
                    continue
                entry = registry.find_by_id(h.catalog_id)
                hits.append({
                    "sha256": h.sha256,
                    "score": h.score,
                    "filename": Path(row["current_path"]).name,
                    "catalog_id": h.catalog_id,
                    "catalog_label": entry.label if entry else "",
                })

        # Multi-catalog mode is "active" when more than one catalog is enabled;
        # controls whether result cards show a per-catalog badge.
        enabled_count = sum(1 for e in registry.entries() if e.enabled and e.available)

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "default_image_weight": image_weight,
                "query": q,
                "hits": hits,
                "registry": registry,
                "multi_catalog": enabled_count > 1,
                "notice": notice,
            },
        )

    @app.get("/thumb/{catalog_id}/{sha256}")
    def thumb(catalog_id: str, sha256: str, size: str = "medium") -> FileResponse:
        try:
            thumb_size = ThumbSize(size)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"unknown size {size!r}")

        catalogs = app.state.catalogs
        thumbs = app.state.thumbs_by_catalog
        resolvers = app.state.path_resolvers

        cat = catalogs.get(catalog_id)
        if cat is None:
            raise HTTPException(status_code=404, detail=f"unknown catalog {catalog_id!r}")
        row = cat.get_photo(sha256)
        if row is None or row["current_path"] is None:
            raise HTTPException(status_code=404, detail=f"no photo for sha {sha256!r}")

        source = resolvers[catalog_id].resolve(row["current_path"])
        if not source.exists():
            raise HTTPException(status_code=404, detail=f"source missing on disk: {source}")

        path = thumbs[catalog_id].get_or_create(sha256, source, thumb_size)
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/photo/{catalog_id}/{sha256}", response_class=HTMLResponse)
    def photo(catalog_id: str, sha256: str, request: Request) -> HTMLResponse:
        catalogs = app.state.catalogs
        cat = catalogs.get(catalog_id)
        if cat is None:
            raise HTTPException(status_code=404, detail=f"unknown catalog {catalog_id!r}")
        row = cat.get_photo(sha256)
        if row is None:
            raise HTTPException(status_code=404, detail=f"no photo for sha {sha256!r}")

        tags = cat.get_tags(sha256)
        return app.state.templates.TemplateResponse(
            request,
            "photo.html",
            {
                "catalog_id": catalog_id,
                "sha256": sha256,
                "filename": Path(row["current_path"]).name if row["current_path"] else "?",
                "caption": row["caption"],
                "tags": [t.name for t in tags],
            },
        )

    @app.get("/catalogs/browse")
    def browse_dirs(path: str | None = None) -> dict:
        from pixsage.discovery import safe_is_dir

        base = Path(path).expanduser() if path else Path.home()
        try:
            base = base.resolve()
        except OSError:
            raise HTTPException(status_code=400, detail=f"bad path: {path}")
        if not safe_is_dir(base):
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

    @app.post("/catalogs/add-scan")
    def add_scan(path: str = Form(...)) -> RedirectResponse:
        from urllib.parse import quote

        from pixsage import discovery
        from pixsage.registry import derive_signatures

        registry = app.state.registry
        root = Path(path).expanduser()
        if not discovery.safe_is_dir(root):
            raise HTTPException(status_code=400, detail=f"not a directory: {root}")

        found_paths = discovery.walk_for_photoindex([root])
        found = len(found_paths)
        added = 0
        for pi in found_paths:
            pi = Path(pi)
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

        skipped = found - added
        if found == 0:
            msg = f"No indexed catalogs found under {root} — nothing added"
        elif skipped == 0:
            msg = f"Added {added} catalog(s)"
        elif added == 0:
            msg = f"All {skipped} catalog(s) under {root} already registered"
        else:
            msg = f"Added {added}; {skipped} already registered"

        return RedirectResponse(url=f"/?notice={quote(msg)}", status_code=303)

    @app.post("/catalogs/{catalog_id}/remove")
    def remove_catalog(catalog_id: str) -> RedirectResponse:
        registry = app.state.registry
        multi = app.state.multi_search
        entry = registry.find_by_id(catalog_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"unknown catalog {catalog_id!r}")
        multi.remove_catalog(catalog_id)
        app.state.catalogs.pop(catalog_id, None)
        app.state.path_resolvers.pop(catalog_id, None)
        app.state.thumbs_by_catalog.pop(catalog_id, None)
        app.state.photoindex_paths.pop(catalog_id, None)
        registry.remove(catalog_id)
        registry.save()
        return RedirectResponse(url="/", status_code=303)

    @app.post("/catalogs/{catalog_id}/rename")
    def rename_catalog(catalog_id: str, label: str = Form(...)) -> RedirectResponse:
        registry = app.state.registry
        entry = registry.find_by_id(catalog_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"unknown catalog {catalog_id!r}")
        registry.rename(catalog_id, label)
        registry.save()
        return RedirectResponse(url="/", status_code=303)

    @app.post("/catalogs/{catalog_id}/toggle")
    def toggle_catalog(catalog_id: str) -> RedirectResponse:
        registry = app.state.registry
        multi = app.state.multi_search
        entry = registry.find_by_id(catalog_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"unknown catalog {catalog_id!r}")
        registry.toggle(catalog_id)
        registry.save()
        # Reload the MultiSearchService entry to match the new enabled state.
        if entry.enabled and entry.available:
            _load_catalog_into_multi(app, entry)
        else:
            multi.remove_catalog(catalog_id)
        return RedirectResponse(url="/", status_code=303)

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

    @app.get("/similar/{catalog_id}/{sha256}", response_class=HTMLResponse)
    def similar(catalog_id: str, sha256: str, request: Request) -> HTMLResponse:
        config = app.state.config
        templates = app.state.templates
        multi = app.state.multi_search
        registry = app.state.registry
        catalogs = app.state.catalogs

        cat = catalogs.get(catalog_id)
        if cat is None:
            raise HTTPException(status_code=404, detail=f"unknown catalog {catalog_id!r}")
        row = cat.get_photo(sha256)
        if row is None:
            raise HTTPException(status_code=404, detail=f"no photo for sha {sha256!r}")

        raw_hits = multi.search_by_image(
            catalog_id=catalog_id, sha256=sha256, top_k=config.search.top_k
        )
        hits = []
        for h in raw_hits:
            # Defensive: search_by_image already excludes the query photo, but
            # filtering here keeps the route honest if that ever changes.
            if h.sha256 == sha256:
                continue
            r = cat.get_photo(h.sha256)
            if r is None:
                continue
            entry = registry.find_by_id(h.catalog_id)
            hits.append({
                "sha256": h.sha256,
                "score": h.score,
                "filename": Path(r["current_path"]).name,
                "catalog_id": h.catalog_id,
                "catalog_label": entry.label if entry else "",
            })

        filename = Path(row["current_path"]).name if row["current_path"] else "?"
        enabled_count = sum(1 for e in registry.entries() if e.enabled and e.available)

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "default_image_weight": config.search.default_image_weight,
                "query": "",
                "hits": hits,
                "similar_to": {
                    "sha256": sha256,
                    "catalog_id": catalog_id,
                    "filename": filename,
                },
                "registry": registry,
                "multi_catalog": enabled_count > 1,
            },
        )

    # ─────────────────────────────────────────────────────────────────────
    # EXPERIMENTAL: HITL cluster-based location labelling.
    #
    # Why this exists:
    #   GeoCLIP failed on the Antarctic α7c corpus (0/1509 predictions in
    #   the actual region) because YFCC100M is biased away from polar
    #   training data. The HITL approach was: cluster photos by visual
    #   similarity, surface clusters to the photographer, let one click
    #   propagate a (lat, lon, place_name) label to every photo in the
    #   cluster.
    #
    # Why it's disabled by default:
    #   The photographer indicated this isn't a workflow they reach for;
    #   they organize by folder structure and don't query by location. The
    #   feature isn't load-bearing for the search / tag / Lightroom flow.
    #
    # Why the code stays:
    #   The clusters module + write_gps / read_gps / user_locations table
    #   are useful infrastructure regardless. If we land on a better
    #   location-labelling UX (interactive map picker, smarter cluster
    #   suggestions) we'd reuse this scaffolding. If we don't, deleting
    #   this whole block (and the corresponding templates / tests / catalog
    #   table) is a one-commit operation later.
    #
    # Single-catalog assumption:
    #   Clustering is per-catalog; the experimental routes pick the first
    #   loaded catalog. Real multi-catalog cluster labelling would need a
    #   /explore/{catalog_id} shape — out of scope while the feature is off.
    #
    # Enable for testing/exploration: build_app(..., experimental_cluster_labelling=True).
    # ─────────────────────────────────────────────────────────────────────
    if not experimental_cluster_labelling:
        return

    def _first_catalog():
        """Return (catalog_id, Catalog, photoindex_path) for the first loaded
        catalog, or raise 404. Cluster routes are inherently single-catalog."""
        catalogs = app.state.catalogs
        if not catalogs:
            raise HTTPException(status_code=404, detail="no catalogs loaded")
        cid = next(iter(catalogs))
        return cid, catalogs[cid], app.state.photoindex_paths[cid]

    @app.get("/explore", response_class=HTMLResponse)
    def explore(request: Request) -> HTMLResponse:
        """Cluster grid for HITL location labelling. Experimental — see comment above."""
        _, catalog, _ = _first_catalog()
        clusters = _get_or_compute_clusters(app)
        cluster_views = []
        for c in clusters:
            label = _cluster_label_summary(c, catalog)
            cluster_views.append({
                "id": c.cluster_id,
                "size": c.size,
                "sample_shas": c.sample_shas,
                "dominant_folder": c.dominant_folder,
                "folder_purity": c.folder_purity,
                "distinctive_tags": c.distinctive_tags[:4],
                "label": label,
            })
        return app.state.templates.TemplateResponse(
            request, "explore.html", {"clusters": cluster_views}
        )

    @app.get("/cluster/{cluster_id}", response_class=HTMLResponse)
    def cluster_detail(request: Request, cluster_id: int) -> HTMLResponse:
        _, catalog, _ = _first_catalog()
        clusters = _get_or_compute_clusters(app)
        cluster = next((c for c in clusters if c.cluster_id == cluster_id), None)
        if cluster is None:
            raise HTTPException(status_code=404, detail=f"no cluster {cluster_id}")
        label = _cluster_label_summary(cluster, catalog)
        return app.state.templates.TemplateResponse(
            request,
            "cluster.html",
            {
                "cluster_id": cluster.cluster_id,
                "size": cluster.size,
                "member_shas": cluster.member_shas,
                "folder_distribution": cluster.folder_distribution,
                "distinctive_tags": cluster.distinctive_tags,
                "label": label,
            },
        )

    @app.post("/cluster/{cluster_id}/label")
    def cluster_label(
        cluster_id: int,
        latitude: float = Form(...),
        longitude: float = Form(...),
        place_name: str = Form(""),
    ) -> RedirectResponse:
        """Apply a (lat, lon, place_name) label to every photo in this cluster.
        Writes XMP GPS + IPTC sublocation, records in user_locations table."""
        from pixsage.xmp import needs_sidecar, write_gps
        cat_id, catalog, _ = _first_catalog()
        resolver = app.state.path_resolvers[cat_id]
        clusters = _get_or_compute_clusters(app)
        cluster = next((c for c in clusters if c.cluster_id == cluster_id), None)
        if cluster is None:
            raise HTTPException(status_code=404, detail=f"no cluster {cluster_id}")
        place = place_name.strip() or None
        applied_via = f"cluster:{cluster_id}"
        for sha in cluster.member_shas:
            row = catalog.get_photo(sha)
            if row is None or not row["current_path"]:
                continue
            path = resolver.resolve(row["current_path"])
            if not path.exists():
                continue
            try:
                write_gps(path, latitude, longitude, place, is_raw=needs_sidecar(path))
            except Exception as e:
                # Don't bail mid-cluster; record what we can in catalog regardless.
                import sys
                print(f"  GPS write failed for {path.name}: {e}", file=sys.stderr)
            catalog.record_user_location(
                sha, latitude, longitude, place, applied_via
            )
        return RedirectResponse(f"/cluster/{cluster_id}", status_code=303)


def _load_catalog_into_multi(app, entry) -> None:
    """Load a single catalog into the MultiSearchService. Used by toggle, add-scan, and refresh."""
    from pixsage.catalog import Catalog
    from pixsage.path_translation import PathResolver
    from pixsage.search import SearchService
    from pixsage.vectors import VectorStore
    from pixsage.web.thumbs import ThumbnailCache

    photoindex = Path(entry.photoindex_path)
    catalog = Catalog(photoindex / "catalog.db")
    catalog.init_schema()
    app.state.catalogs[entry.id] = catalog
    app.state.photoindex_paths[entry.id] = photoindex
    stored_root = catalog.get_meta("photo_root_at_embed")
    app.state.path_resolvers[entry.id] = PathResolver(
        stored_root=stored_root,
        runtime_root=photoindex.parent,
    )
    app.state.thumbs_by_catalog[entry.id] = ThumbnailCache(photoindex / "thumbs")

    vectors = VectorStore(photoindex / "vectors")
    service = SearchService(
        store=vectors,
        embedder=app.state.embedder,
        image_kind=app.state.embedder.info.image_kind,
        text_kind=app.state.embedder.info.text_kind,
    )
    service.load()
    app.state.multi_search.add_catalog(
        catalog_id=entry.id,
        service=service,
        image_sig=entry.image_embedder_signature or DEFAULT_IMAGE_SIGNATURE,
        caption_sig=entry.caption_embedder_signature or DEFAULT_CAPTION_SIGNATURE,
    )


def _get_or_compute_clusters(app):
    """Lazy-compute clusters once per process; cache on app.state.clusters.
    UMAP/HDBSCAN take ~30s on a few thousand photos, so we don't want to do
    this per request."""
    cached = getattr(app.state, "clusters", None)
    if cached is not None:
        return cached

    from pixsage.analysis import load_export
    from pixsage.clusters import compute_clusters

    catalogs = app.state.catalogs
    if not catalogs:
        app.state.clusters = []
        return []
    first_id = next(iter(catalogs))
    photoindex = app.state.photoindex_paths[first_id]
    photo_root = photoindex.parent
    export = load_export(photoindex)
    shas, mats = export.aligned_matrices(require=("image_vec",))
    if len(shas) == 0:
        app.state.clusters = []
        return []
    clusters = compute_clusters(
        shas=list(shas),
        image_matrix=mats["image"],
        photo_paths=export.paths,
        photo_tags=export.tags,
        photo_root=photo_root,
    )
    app.state.clusters = clusters
    return clusters


def _cluster_label_summary(cluster, catalog) -> dict | None:
    """If every photo in this cluster has the same user_location, return it.
    Otherwise return a partial summary or None."""
    locs = []
    for sha in cluster.member_shas:
        loc = catalog.get_user_location(sha)
        if loc is not None:
            locs.append(loc)
    if not locs:
        return None
    if len(locs) < cluster.size:
        return {"partial": True, "labelled": len(locs), "total": cluster.size}
    first = locs[0]
    same = all(
        round(l["latitude"], 4) == round(first["latitude"], 4)
        and round(l["longitude"], 4) == round(first["longitude"], 4)
        for l in locs
    )
    return {
        "partial": False,
        "labelled": len(locs),
        "total": cluster.size,
        "latitude": first["latitude"],
        "longitude": first["longitude"],
        "place_name": first["place_name"],
        "consistent": same,
    }
