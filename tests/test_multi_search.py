from __future__ import annotations
from unittest.mock import MagicMock

from pixsage.multi_search import MultiSearchService, MultiHit
from pixsage.search import Hit


def _fake_search_service(hits_by_query: dict[str, list[tuple[str, float]]]) -> MagicMock:
    """A SearchService stub that returns canned hits per query string."""
    m = MagicMock()
    def search(query, image_weight, top_k):
        return [Hit(sha256=sha, score=score)
                for sha, score in hits_by_query.get(query, [])][:top_k]
    m.search.side_effect = search
    return m


def test_search_merges_results_across_catalogs() -> None:
    sony = _fake_search_service({
        "penguin": [("sha-sony-1", 0.9), ("sha-sony-2", 0.4)],
    })
    iphone = _fake_search_service({
        "penguin": [("sha-iphone-1", 0.7)],
    })
    multi = MultiSearchService()
    multi.add_catalog("cat-sony", sony, image_sig="x", caption_sig="y")
    multi.add_catalog("cat-iphone", iphone, image_sig="x", caption_sig="y")

    hits = multi.search("penguin", image_weight=0.5, top_k=5,
                        query_image_sig="x", query_caption_sig="y")
    # Global merge: 0.9 (sony-1), 0.7 (iphone-1), 0.4 (sony-2)
    assert [h.sha256 for h in hits] == ["sha-sony-1", "sha-iphone-1", "sha-sony-2"]
    assert hits[0].catalog_id == "cat-sony"
    assert hits[1].catalog_id == "cat-iphone"


def test_search_respects_top_k() -> None:
    a = _fake_search_service({
        "q": [("a1", 0.9), ("a2", 0.8), ("a3", 0.7)],
    })
    b = _fake_search_service({
        "q": [("b1", 0.95), ("b2", 0.85)],
    })
    multi = MultiSearchService()
    multi.add_catalog("a", a, image_sig="x", caption_sig="y")
    multi.add_catalog("b", b, image_sig="x", caption_sig="y")

    hits = multi.search("q", image_weight=0.5, top_k=3,
                        query_image_sig="x", query_caption_sig="y")
    assert len(hits) == 3
    assert [h.sha256 for h in hits] == ["b1", "a1", "b2"]


def test_search_skips_catalogs_with_mismatched_signature() -> None:
    """A catalog whose signature doesn't match the query encoder is skipped."""
    sony = _fake_search_service({"q": [("sony-1", 0.9)]})
    iphone = _fake_search_service({"q": [("iphone-1", 0.95)]})
    multi = MultiSearchService()
    multi.add_catalog("sony", sony, image_sig="siglip2@v1", caption_sig="minilm@v2")
    multi.add_catalog("iphone", iphone, image_sig="siglip2@v1", caption_sig="OLD_MINILM")

    hits = multi.search("q", image_weight=0.5, top_k=5,
                        query_image_sig="siglip2@v1", query_caption_sig="minilm@v2")
    catalogs = {h.catalog_id for h in hits}
    assert "sony" in catalogs


def test_search_returns_empty_when_no_catalogs() -> None:
    multi = MultiSearchService()
    hits = multi.search("anything", image_weight=0.5, top_k=5,
                        query_image_sig="x", query_caption_sig="y")
    assert hits == []


def test_search_by_image_delegates_to_owning_catalog() -> None:
    sony = MagicMock()
    sony.search_by_image.return_value = [Hit(sha256="other-sha", score=0.8)]
    iphone = MagicMock()
    iphone.search_by_image.return_value = [Hit(sha256="wrong", score=0.99)]

    multi = MultiSearchService()
    multi.add_catalog("sony", sony, image_sig="x", caption_sig="y")
    multi.add_catalog("iphone", iphone, image_sig="x", caption_sig="y")

    hits = multi.search_by_image(catalog_id="sony", sha256="query-sha", top_k=5)
    sony.search_by_image.assert_called_once_with(sha256="query-sha", top_k=5)
    iphone.search_by_image.assert_not_called()
    assert len(hits) == 1
    assert hits[0].catalog_id == "sony"
    assert hits[0].sha256 == "other-sha"


def test_search_by_image_unknown_catalog_returns_empty() -> None:
    multi = MultiSearchService()
    multi.add_catalog("sony", MagicMock(), image_sig="x", caption_sig="y")
    hits = multi.search_by_image(catalog_id="nonexistent", sha256="x", top_k=5)
    assert hits == []
