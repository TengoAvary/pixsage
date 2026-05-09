from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from pixsage.web.thumbs import ThumbSize


def register(app: FastAPI) -> None:
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
