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
