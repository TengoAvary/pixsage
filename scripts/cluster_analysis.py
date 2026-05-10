"""UMAP + HDBSCAN clustering of a pixsage export.

Side-by-side cluster analysis in the image-similarity space (SigLIP2) and the
caption-similarity space (MiniLM). For each cluster, reports the distinctive
tags (high in-cluster frequency × cluster-specific) and the folder
distribution — useful for HITL labelling, since folder-pure clusters are the
ones a single photographer-supplied label can propagate to safely.

Usage:
    pip install -e ".[search,dashboard]" && pip install umap-learn hdbscan
    python scripts/cluster_analysis.py /path/to/photo_root/.photoindex

Requires: umap-learn, hdbscan, scikit-learn.
"""
import argparse
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from collections import Counter
from pathlib import Path

import umap
import hdbscan
from sklearn.metrics import adjusted_rand_score

from pixsage.analysis import load_export

parser = argparse.ArgumentParser()
parser.add_argument("photoindex", type=Path, help="Path to a .photoindex/ directory.")
parser.add_argument("--min-cluster-size", type=int, default=15)
parser.add_argument("--top-n-display", type=int, default=12,
                    help="How many clusters to print per space.")
args = parser.parse_args()

EXPORT = args.photoindex
e = load_export(EXPORT)
shas, mats = e.aligned_matrices(require=("image_vec", "caption_vec"))
print(f"Aligned: {len(shas)} photos with both image and caption vectors")
print(f"  image:  {mats['image'].shape}")
print(f"  caption: {mats['caption'].shape}")

print("\nRunning UMAP on image space → 30D (cosine metric)...")
img_emb = umap.UMAP(n_components=30, n_neighbors=15, min_dist=0.0,
                    metric="cosine", random_state=42).fit_transform(mats["image"])

print("Running UMAP on caption space → 30D (cosine metric)...")
cap_emb = umap.UMAP(n_components=30, n_neighbors=15, min_dist=0.0,
                    metric="cosine", random_state=42).fit_transform(mats["caption"])

print("\nRunning HDBSCAN on each...")
img_labels = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size, min_samples=5).fit_predict(img_emb)
cap_labels = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size, min_samples=5).fit_predict(cap_emb)


def folder_of(path):
    parts = path.replace("\\", "/").split("/")
    try:
        i = parts.index("Sony alpha 7c")
        return parts[i + 1] if i + 1 < len(parts) else "?"
    except ValueError:
        return "?"


def cluster_summary(name, labels, shas, e, top_n=8):
    counter = Counter(labels)
    n_clusters = sum(1 for k in counter if k != -1)
    n_noise = counter.get(-1, 0)
    print(f"\n=== {name} CLUSTERS ===")
    print(f"clusters: {n_clusters}, noise points: {n_noise} ({100*n_noise/len(labels):.0f}%)")

    # Compute global tag frequencies for tf-idf-style cluster labelling
    all_tags = Counter()
    for s in shas:
        all_tags.update(e.tags.get(s, []))

    sorted_clusters = sorted(
        (c for c in counter if c != -1), key=lambda c: -counter[c]
    )
    for cid in sorted_clusters[:top_n]:
        members = [shas[i] for i in range(len(labels)) if labels[i] == cid]
        # Find tags that are common WITHIN cluster but not globally
        cluster_tags = Counter()
        for s in members:
            cluster_tags.update(e.tags.get(s, []))
        # tf-idf-ish: tag-in-cluster-fraction × log(N / global_count)
        total = len(shas)
        scored_tags = [
            (t, c, c / len(members), c / max(1, all_tags[t]))
            for t, c in cluster_tags.most_common()
            if c >= 3
        ]
        # Sort by (in-cluster fraction × log of cluster-specificity)
        scored_tags.sort(key=lambda x: -(x[2] * x[3]))
        top_tags = [f"{t} ({c}/{len(members)})" for t, c, _, _ in scored_tags[:6]]

        # Folder distribution
        folder_counts = Counter(folder_of(e.paths[s]) for s in members)
        folder_str = ", ".join(f"{f}={n}" for f, n in folder_counts.most_common(3))

        # Sample 3 captions
        captions = [e.captions.get(s, "")[:55] for s in members[:3] if s in e.captions]

        print(f"\n  cluster {cid} (n={counter[cid]}):")
        print(f"    distinctive tags: {', '.join(top_tags)}")
        print(f"    folders: {folder_str}")
        for cap in captions:
            print(f"    · {cap!r}")


cluster_summary("IMAGE", img_labels, shas, e, top_n=args.top_n_display)
cluster_summary("CAPTION", cap_labels, shas, e, top_n=args.top_n_display)

ari = adjusted_rand_score(img_labels, cap_labels)
print(f"\nADJUSTED RAND INDEX (image vs caption): {ari:.4f}")
print("  0 = random agreement; 1 = identical clusters; ~0.05-0.2 typical for "
      "different views; >0.4 means strong alignment.")

# Cross-tab: how do image clusters break down by caption cluster?
img_set = sorted(set(img_labels) - {-1})
cap_set = sorted(set(cap_labels) - {-1})
print(f"\nFor each major image cluster, top caption-cluster overlap:")
for img_c in [c for c in img_set if Counter(img_labels)[c] >= 30][:5]:
    members_idx = [i for i in range(len(img_labels)) if img_labels[i] == img_c]
    cap_dist = Counter(cap_labels[i] for i in members_idx)
    n = len(members_idx)
    top_cap_cluster, top_cap_count = max(cap_dist.items(), key=lambda kv: kv[1])
    label = f"caption #{top_cap_cluster}" if top_cap_cluster != -1 else "noise"
    print(f"  image cluster {img_c} (n={n}): {label} got {top_cap_count}/{n} ({100*top_cap_count/n:.0f}%)")
