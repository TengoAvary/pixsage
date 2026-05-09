from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse


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
