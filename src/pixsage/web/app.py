from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pixsage.catalog import Catalog
from pixsage.config import load_config, ensure_default_config
from pixsage.search import SearchService
from pixsage.vectors import VectorStore


WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"


def build_app(photo_root: Path, embedder_name: str = "siglip2") -> FastAPI:
    """Construct the FastAPI app for a photo root.

    Loads catalog, vectors, and the search service eagerly so route handlers
    can stay synchronous and stateless.
    """
    photoindex = photo_root / ".photoindex"
    photoindex.mkdir(exist_ok=True)
    catalog_path = photoindex / "catalog.db"
    cfg_path = photoindex / "vocabulary.toml"
    ensure_default_config(cfg_path)
    config = load_config(cfg_path)

    catalog = Catalog(catalog_path)
    catalog.init_schema()

    # Lazy import to keep `pixsage embed` callable on systems without [search] installed.
    from pixsage.cli import _build_embedder
    from pixsage.device import select_device
    embedder = _build_embedder(embedder_name)
    embedder.load(select_device())

    vectors = VectorStore(photoindex / "vectors")
    search_service = SearchService(
        store=vectors,
        embedder=embedder,
        image_kind=embedder.info.image_kind,
        text_kind=embedder.info.text_kind,
    )
    search_service.load()

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    app = FastAPI(title="pixsage")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Stash everything on the app for routes (Task 13+) to grab.
    app.state.photo_root = photo_root
    app.state.config = config
    app.state.catalog = catalog
    app.state.vectors = vectors
    app.state.embedder = embedder
    app.state.search = search_service
    app.state.templates = templates

    from pixsage.web import routes
    routes.register(app)

    return app
