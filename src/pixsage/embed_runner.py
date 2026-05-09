from __future__ import annotations

from pathlib import Path

import numpy as np

from pixsage.catalog import Catalog
from pixsage.embedders.base import Embedder
from pixsage.images import load_image
from pixsage.vectors import VectorStore


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

        rows = list(self.catalog.iter_photos_for_embedding())
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
                    txt_vec = self.embedder.embed_text([caption])[0]
                    self.vectors.append(info.text_kind, [(sha, txt_vec)])

                stats["processed"] += 1
            except Exception as e:
                self.catalog.mark_error(sha, str(e))
                stats["errored"] += 1

        return stats

    def _caption_is_stale(self, kind: str, sha: str, caption_updated_at: str | None) -> bool:
        if caption_updated_at is None:
            return False
        vec_ts = self.vectors.created_at(kind, sha)
        if vec_ts is None:
            return True
        return caption_updated_at > vec_ts
