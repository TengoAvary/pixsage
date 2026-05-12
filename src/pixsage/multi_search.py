"""Orchestrator that holds N per-catalog SearchServices and merges results.

Each per-catalog SearchService is the existing class from pixsage.search.
The orchestrator's job is purely to:
1. Track which catalogs are loaded and what their embedder signatures are.
2. Route queries to compatible catalogs.
3. Merge per-catalog top-k results into a global top-k, preserving catalog_id
   for UI badging.

Encoder compatibility:
- For text queries, the orchestrator decides per-catalog whether to ask: any
  catalog whose image-signature matches the query's image encoder OR whose
  caption-signature matches the query's caption encoder will be asked. The
  per-catalog SearchService's own blend logic then handles partial-channel
  cases (a catalog missing one channel scores it as 0 — see search.py).
"""
from __future__ import annotations

from dataclasses import dataclass

from pixsage.search import Hit, SearchService


@dataclass(frozen=True)
class MultiHit:
    """A search hit, augmented with the source catalog id."""
    sha256: str
    score: float
    catalog_id: str


@dataclass
class _CatalogSlot:
    service: SearchService
    image_sig: str
    caption_sig: str


class MultiSearchService:
    def __init__(self) -> None:
        self._catalogs: dict[str, _CatalogSlot] = {}

    def add_catalog(
        self,
        catalog_id: str,
        service: SearchService,
        image_sig: str,
        caption_sig: str,
    ) -> None:
        self._catalogs[catalog_id] = _CatalogSlot(
            service=service, image_sig=image_sig, caption_sig=caption_sig,
        )

    def remove_catalog(self, catalog_id: str) -> None:
        self._catalogs.pop(catalog_id, None)

    def catalog_ids(self) -> list[str]:
        return list(self._catalogs.keys())

    def search(
        self,
        query: str,
        image_weight: float,
        top_k: int,
        query_image_sig: str,
        query_caption_sig: str,
    ) -> list[MultiHit]:
        """Run the query across all compatible catalogs; merge by score."""
        if not self._catalogs:
            return []

        all_hits: list[MultiHit] = []
        for cat_id, slot in self._catalogs.items():
            # A catalog participates if it matches the query on at least one
            # channel. Per-channel skip is handled inside the per-catalog
            # SearchService via the existing 0-score fallback for missing
            # channels, but if BOTH channels mismatch, skip the catalog.
            if (
                slot.image_sig != query_image_sig
                and slot.caption_sig != query_caption_sig
            ):
                continue
            per_cat_hits = slot.service.search(
                query=query, image_weight=image_weight, top_k=top_k,
            )
            for h in per_cat_hits:
                all_hits.append(
                    MultiHit(sha256=h.sha256, score=h.score, catalog_id=cat_id)
                )

        all_hits.sort(key=lambda h: h.score, reverse=True)
        return all_hits[:top_k]
