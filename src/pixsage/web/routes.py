from __future__ import annotations

from fastapi import FastAPI, Request
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
