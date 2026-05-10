"""Visual-similarity clustering of a pixsage corpus.

Clusters are computed via UMAP (cosine, → 30D) followed by HDBSCAN. The result
is a list of `Cluster` summaries — sample shas (closest-to-medoid), folder
dominance, and member sha sets.

Production use today: `scripts/cluster_analysis.py` runs this offline against
an exported `.photoindex/` to characterize a corpus's visual structure.

Was also wired into the experimental HITL location-labelling UI in the
webapp (currently disabled by default — see pixsage/web/routes.py for
context). Kept here as a reusable building block in case we re-enable that
flow or build a different cluster-driven feature.

Compute is heavy (~30s for ~1500 photos); callers are expected to cache.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class Cluster:
    cluster_id: int
    member_shas: list[str]                # in distance-from-medoid order
    sample_shas: list[str]                # first 4 of member_shas
    folder_distribution: list[tuple[str, int]]  # [(folder_name, count), …] descending
    distinctive_tags: list[str]           # tags common in cluster, rare globally
    size: int = field(init=False)

    def __post_init__(self) -> None:
        self.size = len(self.member_shas)

    @property
    def dominant_folder(self) -> tuple[str, int] | None:
        return self.folder_distribution[0] if self.folder_distribution else None

    @property
    def folder_purity(self) -> float:
        if not self.folder_distribution or not self.size:
            return 0.0
        return self.folder_distribution[0][1] / self.size


def compute_clusters(
    shas: list[str],
    image_matrix: np.ndarray,
    photo_paths: dict[str, str],
    photo_tags: dict[str, list[str]],
    photo_root: Path,
    *,
    min_cluster_size: int = 15,
) -> list[Cluster]:
    """Return clusters in descending size order. Photos in HDBSCAN noise
    (label = -1) are dropped — they don't form a useful cluster for labelling."""
    import hdbscan
    import umap

    if len(shas) < min_cluster_size:
        return []

    embedded = umap.UMAP(
        n_components=30,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    ).fit_transform(image_matrix)

    labels = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=5
    ).fit_predict(embedded)

    # Global tag frequency for tf-idf-style cluster characterization
    all_tags: Counter[str] = Counter()
    for s in shas:
        all_tags.update(photo_tags.get(s, []))

    clusters: list[Cluster] = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        member_idx = np.where(labels == cid)[0]
        members = [shas[i] for i in member_idx]

        # Order members by closeness to medoid (mean of cluster's UMAP embeddings)
        cluster_emb = embedded[member_idx]
        centroid = cluster_emb.mean(axis=0)
        distances = np.linalg.norm(cluster_emb - centroid, axis=1)
        ordered = [members[i] for i in np.argsort(distances)]

        # Folder distribution (top-level folder under photo_root)
        folder_counts = Counter(
            _top_level_folder(photo_paths[s], photo_root) for s in members
        )
        folder_distribution = folder_counts.most_common()

        # Distinctive tags: high in-cluster fraction × cluster-specific share
        cluster_tags: Counter[str] = Counter()
        for s in members:
            cluster_tags.update(photo_tags.get(s, []))
        scored = [
            (
                tag,
                count / len(members) * (count / max(1, all_tags[tag])),
                count,
            )
            for tag, count in cluster_tags.items()
            if count >= max(3, len(members) // 10)
        ]
        scored.sort(key=lambda x: -x[1])
        distinctive_tags = [t for t, _, _ in scored[:6]]

        clusters.append(
            Cluster(
                cluster_id=int(cid),
                member_shas=ordered,
                sample_shas=ordered[:4],
                folder_distribution=folder_distribution,
                distinctive_tags=distinctive_tags,
            )
        )

    clusters.sort(key=lambda c: -c.size)
    return clusters


def _top_level_folder(photo_path: str, photo_root: Path) -> str:
    """First path segment under photo_root.

    For photo_root='E:/Sony alpha 7c' and photo_path='E:\\Sony alpha 7c\\Seymour\\Fieldwork\\DSC.ARW',
    returns 'Seymour'.
    """
    try:
        rel = Path(photo_path).resolve().relative_to(Path(photo_root).resolve())
    except (ValueError, OSError):
        return Path(photo_path).parent.name or "?"
    return rel.parts[0] if rel.parts else "?"
