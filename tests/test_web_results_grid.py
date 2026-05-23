from __future__ import annotations

import types
from pathlib import Path

import jinja2


def _templates_dir() -> Path:
    """Return the absolute path to the web templates directory."""
    return Path(__file__).parent.parent / "src" / "pixsage" / "web" / "templates"


def _fake_hit(n: int) -> types.SimpleNamespace:
    """Return a minimal hit namespace that satisfies _card.html's accessors."""
    return types.SimpleNamespace(
        sha256=f"{'a' * 64}",
        catalog_id=n,
        filename=f"photo_{n}.jpg",
        score=0.9 - n * 0.1,
        catalog_label=f"Catalog {n}",
    )


def test_results_partial_does_not_emit_grid_div() -> None:
    """The `_results.html` partial must not wrap cards in `class="grid"`.

    `style.css` grids the parent `<section id="results">`; a nested
    `.grid` div would break the layout (the parent grid sees one full-
    width child instead of the cards).

    This test renders the partial directly via Jinja2 with a non-empty
    ``hits`` list so the ``{% if hits %}`` branch is exercised and the
    assertion is meaningful — a stray ``<div class="grid">`` inside that
    branch would be caught.
    """
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_templates_dir())),
        autoescape=jinja2.select_autoescape(["html"]),
    )
    hits = [_fake_hit(i) for i in range(3)]
    rendered = env.get_template("_results.html").render(
        hits=hits,
        query="sunset",
        multi_catalog=False,
    )

    # The if-hits branch must be taken (3 hits were passed in).
    assert "photo_0.jpg" in rendered, "hits branch was not rendered"

    # No wrapping .grid div anywhere in the partial.
    assert 'class="grid"' not in rendered
