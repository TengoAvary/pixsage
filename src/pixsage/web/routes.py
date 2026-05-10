from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from pixsage.web.thumbs import ThumbSize


def register(app: FastAPI, *, experimental_cluster_labelling: bool = False) -> None:
    """Register all routes on `app`.

    `experimental_cluster_labelling` (default off) controls the HITL
    cluster-based location labelling routes (/explore, /cluster/{id},
    /cluster/{id}/label). See the comment above the cluster routes below for
    the why-it-exists / why-it's-disabled context.
    """
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        templates = app.state.templates
        config = app.state.config
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "default_image_weight": config.search.default_image_weight,
            },
        )

    @app.post("/search", response_class=HTMLResponse)
    def search(
        request: Request,
        q: str = Form(""),
        image_weight: float = Form(0.5),
    ) -> HTMLResponse:
        templates = app.state.templates
        catalog = app.state.catalog
        config = app.state.config

        if not q.strip():
            return templates.TemplateResponse(
                request,
                "_results.html",
                {"hits": [], "query": q},
            )

        service = app.state.search
        raw_hits = service.search(q, image_weight=image_weight, top_k=config.search.top_k)

        # Enrich each hit with current_path + filename for the card template.
        hits = []
        for h in raw_hits:
            row = catalog.get_photo(h.sha256)
            if row is None:
                continue
            hits.append({
                "sha256": h.sha256,
                "score": h.score,
                "filename": Path(row["current_path"]).name,
            })

        return templates.TemplateResponse(
            request,
            "_results.html",
            {"hits": hits, "query": q},
        )

    @app.get("/thumb/{sha256}")
    def thumb(sha256: str, size: str = "medium") -> FileResponse:
        try:
            thumb_size = ThumbSize(size)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"unknown size {size!r}")

        catalog = app.state.catalog
        row = catalog.get_photo(sha256)
        if row is None or row["current_path"] is None:
            raise HTTPException(status_code=404, detail=f"no photo for sha {sha256!r}")

        source = Path(row["current_path"])
        if not source.exists():
            raise HTTPException(status_code=404, detail=f"source missing on disk: {source}")

        thumbs = app.state.thumbs
        path = thumbs.get_or_create(sha256, source, thumb_size)
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/photo/{sha256}", response_class=HTMLResponse)
    def photo(request: Request, sha256: str) -> HTMLResponse:
        catalog = app.state.catalog
        row = catalog.get_photo(sha256)
        if row is None:
            raise HTTPException(status_code=404, detail=f"no photo for sha {sha256!r}")

        tags = catalog.get_tags(sha256)
        return app.state.templates.TemplateResponse(
            request,
            "photo.html",
            {
                "sha256": sha256,
                "filename": Path(row["current_path"]).name if row["current_path"] else "?",
                "caption": row["caption"],
                "tags": [t.name for t in tags],
            },
        )

    @app.get("/similar/{sha256}", response_class=HTMLResponse)
    def similar(request: Request, sha256: str) -> HTMLResponse:
        catalog = app.state.catalog
        config = app.state.config
        templates = app.state.templates
        service = app.state.search

        if catalog.get_photo(sha256) is None:
            raise HTTPException(status_code=404, detail=f"no photo for sha {sha256!r}")

        raw_hits = service.search_by_image(sha256, top_k=config.search.top_k)
        hits = []
        for h in raw_hits:
            # Exclude the query photo itself
            if h.sha256 == sha256:
                continue
            row = catalog.get_photo(h.sha256)
            if row is None:
                continue
            hits.append({
                "sha256": h.sha256,
                "score": h.score,
                "filename": Path(row["current_path"]).name,
            })

        return templates.TemplateResponse(
            request,
            "_results.html",
            {"hits": hits, "query": "similar images"},
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
    # Enable for testing/exploration: build_app(..., experimental_cluster_labelling=True).
    # ─────────────────────────────────────────────────────────────────────
    if not experimental_cluster_labelling:
        return

    @app.get("/explore", response_class=HTMLResponse)
    def explore(request: Request) -> HTMLResponse:
        """Cluster grid for HITL location labelling. Experimental — see comment above."""
        clusters = _get_or_compute_clusters(app)
        catalog = app.state.catalog
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
        clusters = _get_or_compute_clusters(app)
        catalog = app.state.catalog
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
        clusters = _get_or_compute_clusters(app)
        catalog = app.state.catalog
        cluster = next((c for c in clusters if c.cluster_id == cluster_id), None)
        if cluster is None:
            raise HTTPException(status_code=404, detail=f"no cluster {cluster_id}")
        place = place_name.strip() or None
        applied_via = f"cluster:{cluster_id}"
        for sha in cluster.member_shas:
            row = catalog.get_photo(sha)
            if row is None or not row["current_path"]:
                continue
            path = Path(row["current_path"])
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


def _get_or_compute_clusters(app):
    """Lazy-compute clusters once per process; cache on app.state.clusters.
    UMAP/HDBSCAN take ~30s on a few thousand photos, so we don't want to do
    this per request."""
    cached = getattr(app.state, "clusters", None)
    if cached is not None:
        return cached

    from pixsage.analysis import load_export
    from pixsage.clusters import compute_clusters

    photo_root = app.state.photo_root
    export = load_export(photo_root / ".photoindex")
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
