"""
Clustering Engine Module

Groups similar item embeddings using K-Means clustering.
Enables per-category counting.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : ClusteringEngine (this file)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <ClusteringEngine>  → K-Means based clustering                           │
  │  <ClusterInfo>       → Per-cluster metadata                               │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <sklearn.cluster.KMeans>  ← from sklearn (clustering algorithm)          │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : numpy.ndarray, int, str

Production Rules:
  ClusteringEngine → __init__ + fit + predict + get_cluster_counts
═══════════════════════════════════════════════════════════════════════════════

Pattern: Strategy
- Clustering algorithm can be swapped (K-Means, DBSCAN, etc.)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import Counter


@dataclass
class ClusterInfo:
    """Information about a cluster."""

    cluster_id: int
    count: int
    centroid: np.ndarray
    label: str = "Unknown"  # Optional label from VLM
    representative_idx: int = -1  # Index of most representative member


class ClusteringEngine:
    """
    DBSCAN based clustering for grouping similar items.
    Automatically detects number of clusters based on density.
    Tracks clusters across frames using centroid matching.
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 3,
        centroid_match_threshold: float = 0.85,  # Higher threshold for DINOv2 (more precise)
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.match_threshold = centroid_match_threshold

        # Persistent cluster centroids for tracking
        self.known_centroids: List[np.ndarray] = []
        self.cluster_labels: List[str] = []
        self.cluster_history: Dict[int, List[int]] = {}  # cluster_id -> count history

    def fit_predict(
        self, embeddings: np.ndarray, n_clusters: Optional[int] = None
    ) -> Tuple[np.ndarray, List[ClusterInfo]]:
        """
        Cluster embeddings using DBSCAN and assign persistent IDs.

        Args:
            embeddings: (N, D) array of embeddings
            n_clusters: Ignored (DBSCAN determines this)

        Returns:
            Tuple of (cluster_labels array, list of ClusterInfo)
        """
        if len(embeddings) == 0:
            return np.array([]), []

        # Run DBSCAN
        try:
            from sklearn.cluster import DBSCAN

            # Metric cosine is better for embeddings, but requires 1 - sim distance
            # Or normalize embeddings and use euclidean (equivalent to cosine on unit sphere)
            # Embeddings are already normalized in EmbeddingEngine

            dbscan = DBSCAN(
                eps=self.eps, min_samples=self.min_samples, metric="euclidean"
            )
            labels = dbscan.fit_predict(embeddings)

        except ImportError:
            print(
                "[ClusteringEngine] sklearn not found, using valid fallback requires libraries"
            )
            return np.zeros(len(embeddings), dtype=int), []

        # --- REASONING LAYER: Semantic Merge ---
        # DBSCAN can be too strict. We visually inspect detected clusters and merge them
        # if they look the same (Cosine Similarity > threshold).
        labels = self._merge_similar_clusters(labels, embeddings)

        # Calculate centroids and build cluster info
        unique_labels = set(labels)
        clusters = []

        for label in unique_labels:
            if label == -1:
                # Noise points
                continue

            mask = labels == label
            count = mask.sum()

            # Calculate centroid for this group
            group_embeddings = embeddings[mask]
            centroid = group_embeddings.mean(axis=0)

            # Find representative (closest to centroid)
            distances = np.linalg.norm(group_embeddings - centroid, axis=1)
            rep_local_idx = np.argmin(distances)
            rep_global_idx = np.where(mask)[0][rep_local_idx]

            # Match to known centroids for persistent tracking
            matched_id, matched_label = self._match_to_known(centroid)

            # Assign persistent ID
            persistent_id = (
                matched_id
                if matched_id >= 0
                else self._register_new_cluster(centroid, "Unknown")
            )

            # Update specific label in the return array to the persistent ID
            # NOTE: multiple DBSCAN clusters could map to same persistent ID if they are close
            # But usually DBSCAN merges them.
            # We map local DBSCAN label -> Persistent ID
            # This requires updating the 'labels' array return value
            labels[mask] = persistent_id

            clusters.append(
                ClusterInfo(
                    cluster_id=persistent_id,
                    count=int(count),
                    centroid=centroid,
                    label=matched_label if matched_id >= 0 else "Unknown",
                    representative_idx=int(rep_global_idx),
                )
            )

        # Update known centroids (smoothing)
        self._update_known_centroids(clusters)

        return labels, clusters

    def _merge_similar_clusters(
        self, labels: np.ndarray, embeddings: np.ndarray, threshold: float = 0.85
    ) -> np.ndarray:
        """
        Merge clusters that are semantically similar (cosine similarity > threshold).
        This acts as a 'Reasoning' step to fix over-segmentation.
        """
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        sorted_labels = sorted(list(unique_labels))
        n_labels = len(sorted_labels)

        if n_labels < 2:
            return labels

        # Calculate temporary centroids
        centroids = {}
        for label in sorted_labels:
            mask = labels == label
            centroids[label] = embeddings[mask].mean(axis=0)

        # Find merge pairs
        # Map: from_label -> to_label
        merge_map = {}

        # Greedily merge
        # We iterate and if A and B are similar, we map B -> A (keep A)
        # Note: This is a simple single-pass merge. A graph-based connected component approach would be stricter
        # but this suffices for "merging neighbors".
        for i in range(n_labels):
            label_a = sorted_labels[i]
            if label_a in merge_map:  # Already merged into something else
                continue

            centroid_a = centroids[label_a]

            for j in range(i + 1, n_labels):
                label_b = sorted_labels[j]
                if label_b in merge_map:
                    continue

                centroid_b = centroids[label_b]

                # Cosine Similarity
                sim = np.dot(centroid_a, centroid_b) / (
                    np.linalg.norm(centroid_a) * np.linalg.norm(centroid_b) + 1e-8
                )

                if sim > threshold:
                    # Merge B into A
                    print(
                        f"[Reasoning] Merging Cluster {label_b} -> {label_a} (Similarity: {sim:.3f})"
                    )
                    merge_map[label_b] = label_a

        # Apply merges
        new_labels = labels.copy()
        for src, dst in merge_map.items():
            new_labels[new_labels == src] = dst

        return new_labels

    def _register_new_cluster(self, centroid: np.ndarray, label: str) -> int:
        """Register a new cluster and return its ID."""
        new_id = len(self.known_centroids)
        self.known_centroids.append(centroid.copy())
        self.cluster_labels.append(label)
        return new_id

    def _match_to_known(self, centroid: np.ndarray) -> Tuple[int, str]:
        """Match a centroid to known clusters."""
        if len(self.known_centroids) == 0:
            return -1, "Unknown"

        # Compute similarities
        similarities = []
        for known in self.known_centroids:
            sim = np.dot(centroid, known) / (
                np.linalg.norm(centroid) * np.linalg.norm(known) + 1e-8
            )
            similarities.append(sim)

        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]

        if best_sim >= self.match_threshold:
            return best_idx, self.cluster_labels[best_idx]
        else:
            return -1, "Unknown"

    def _update_known_centroids(self, clusters: List[ClusterInfo]):
        """Update known centroids with new clusters."""
        for cluster in clusters:
            if cluster.cluster_id < len(self.known_centroids):
                # Update existing centroid (moving average)
                alpha = 0.3
                self.known_centroids[cluster.cluster_id] = (
                    alpha * cluster.centroid
                    + (1 - alpha) * self.known_centroids[cluster.cluster_id]
                )
            else:
                # Add new centroid
                self.known_centroids.append(cluster.centroid.copy())
                self.cluster_labels.append(cluster.label)

    def set_cluster_label(self, cluster_id: int, label: str):
        """Assign a label to a cluster (e.g., from VLM)."""
        while len(self.cluster_labels) <= cluster_id:
            self.cluster_labels.append("Unknown")
        self.cluster_labels[cluster_id] = label

    def get_cluster_counts(self, clusters: List[ClusterInfo]) -> Dict[str, int]:
        """Get counts per cluster label."""
        counts = {}
        for cluster in clusters:
            label = (
                cluster.label
                if cluster.label != "Unknown"
                else f"Type {cluster.cluster_id}"
            )
            counts[label] = counts.get(label, 0) + cluster.count
        return counts

    def reset(self):
        """Reset cluster memory."""
        self.known_centroids = []
        self.cluster_labels = []
        self.cluster_history = {}


if __name__ == "__main__":
    # Quick test
    engine = ClusteringEngine()

    # Fake embeddings
    embeddings = np.random.randn(20, 512)
    labels, clusters = engine.fit_predict(embeddings)

    print(f"Labels: {labels}")
    print(f"Clusters: {len(clusters)}")
    for c in clusters:
        print(f"  Cluster {c.cluster_id}: {c.count} items")
