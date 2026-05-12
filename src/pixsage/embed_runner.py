from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

from pixsage.catalog import Catalog
from pixsage.embedders.base import Embedder
from pixsage.images import load_image
from pixsage.registry import DEFAULT_IMAGE_SIGNATURE, DEFAULT_CAPTION_SIGNATURE
from pixsage.vectors import VectorStore
from pixsage.xmp import needs_sidecar, read_xmp


# Florence-2 captions are long descriptive prose ("The image is a close-up of
# a vintage camera resting on a piano keyboard. The camera is black and...").
# Two problems for SigLIP2-style retrieval:
#   1. The "The image is/shows/depicts a..." prefix consumes 4–6 tokens of
#      pure boilerplate, identical across every photo, dragging all caption
#      vectors into the same region of the embedding space.
#   2. SigLIP2's text encoder was trained on short prompts (~5–15 tokens).
#      Long prose past the first sentence (orientation cues like "back to the
#      camera", lighting descriptions, etc.) hurts more than it helps.
# We compress to: prefix-stripped first sentence. Empirically this matches the
# distribution SigLIP2 was trained on and recovers lexical signal that gets
# diluted by the rest of the caption.
_PREFIX_RE = re.compile(
    r"^\s*the image (is|shows|depicts|contains|features|displays|appears to (be|show))\b[^.]*?\b(an?|the|of)\b\s*",
    re.IGNORECASE,
)


def normalize_caption_for_embedding(caption: str) -> str:
    """Strip Florence-2 boilerplate prefix and keep only the first sentence.

    "The image is a close-up of a vintage camera resting on a piano. The
    camera is black..." → "vintage camera resting on a piano".
    """
    text = caption.strip()
    text = _PREFIX_RE.sub("", text, count=1)
    # First sentence (split on period followed by space or end-of-string).
    m = re.search(r"\.\s|\.$", text)
    if m:
        text = text[: m.start()]
    return text.strip() or caption.strip()  # fall back to raw if normalization emptied it


class EmbedRunner:
    """Walks the catalog and computes embeddings for each photo using one embedder.

    For each photo:
      - skip if image-vector already exists (and not --force, and caption isn't stale)
      - load the image, embed it
      - if a caption exists, embed it too
      - write rows to the VectorStore (which dedupes by sha256)
    """

    def __init__(
        self,
        catalog: Catalog,
        vectors: VectorStore,
        embedder: Embedder,
        force: bool = False,
        embed_image: bool = True,
        embed_caption: bool = True,
        progress: bool = False,
    ) -> None:
        self.catalog = catalog
        self.vectors = vectors
        self.embedder = embedder
        self.force = force
        self.embed_image = embed_image
        self.embed_caption = embed_caption
        self.progress = progress

    def run(self) -> dict[str, int]:
        info = self.embedder.info
        stats = {"processed": 0, "skipped": 0, "errored": 0}

        rows = list(self.catalog.iter_photos_for_embedding(include_errored=self.force))
        if self.progress:
            from tqdm import tqdm
            iterator = tqdm(rows, unit="photo")
        else:
            iterator = rows

        for row in iterator:
            sha = row["sha256"]
            current_path = row["current_path"]
            caption = row["caption"]
            caption_updated_at = row["caption_updated_at"]

            # Backfill caption from XMP if catalog row predates Phase 3.
            if caption is None and self.embed_caption:
                try:
                    fields = read_xmp(Path(current_path), is_raw=needs_sidecar(Path(current_path)))
                    if fields.description:
                        self.catalog.record_caption(sha, fields.description)
                        caption = fields.description
                        caption_updated_at = self.catalog.get_photo(sha)["caption_updated_at"]
                except Exception:
                    # XMP read failures shouldn't kill the embed run; we just skip
                    # caption embedding for this photo.
                    pass

            needs_image = self.embed_image and (
                self.force or self.vectors.get_one(info.image_kind, sha) is None
            )
            needs_text = self.embed_caption and caption is not None and (
                self.force
                or self.vectors.get_one(info.text_kind, sha) is None
                or self._caption_is_stale(info.text_kind, sha, caption_updated_at)
            )

            if not needs_image and not needs_text:
                stats["skipped"] += 1
                continue

            try:
                if needs_image:
                    img = load_image(Path(current_path))
                    img_vec = self.embedder.embed_image([img])[0]
                    self.vectors.append(info.image_kind, [(sha, img_vec)])

                if needs_text:
                    txt_vec = self.embedder.embed_caption([normalize_caption_for_embedding(caption)])[0]
                    self.vectors.append(info.text_kind, [(sha, txt_vec)])

                self.catalog.clear_error(sha)
                stats["processed"] += 1
            except Exception as e:
                self.catalog.mark_error(sha, str(e))
                stats["errored"] += 1
                msg = f"  error on {Path(current_path).name}: {e}"
                if self.progress:
                    from tqdm import tqdm
                    tqdm.write(msg, file=sys.stderr)
                else:
                    sys.stderr.write(msg + "\n")

        # Record which embedder produced these vectors so a future cross-catalog
        # search can detect mismatched encoders. TODO(multi-embedder): currently
        # hard-coded to the SigLIP2 + MiniLM defaults — only one embedder ships
        # today, so this is correct in production. Mock-embedder test runs will
        # tag their catalogs with these constants too, which is dishonest but
        # harmless (mock catalogs never cross-search with real-encoder catalogs).
        # When a second embedder is added, source the signature from
        # self.embedder.info instead.
        if stats["processed"] > 0:
            self.catalog.set_meta("image_embedder_signature", DEFAULT_IMAGE_SIGNATURE)
            self.catalog.set_meta("caption_embedder_signature", DEFAULT_CAPTION_SIGNATURE)

        return stats

    def _caption_is_stale(self, kind: str, sha: str, caption_updated_at: str | None) -> bool:
        if caption_updated_at is None:
            return False
        vec_ts = self.vectors.created_at(kind, sha)
        if vec_ts is None:
            return True
        return caption_updated_at > vec_ts
