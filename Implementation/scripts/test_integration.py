"""Integration Test for Vision Pipeline

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : TestIntegration (this module)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <test_integration> → test logic                                          │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <EmbeddingEngine> ← from v2_logic.models.embedding_engine                 │
  │  <ClusteringEngine> ← from v2_logic.models.clustering_engine               │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : str, int, list, np.ndarray

Production Rules:
  TestIntegration → imports + <test_integration>
═══════════════════════════════════════════════════════════════════════════════

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : Script (this file)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <Main>           → Script entry point                                    │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <Imports>        ← External dependencies                                 │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : str, int, etc.

Production Rules:
  Script          → Imports + Main
═══════════════════════════════════════════════════════════════════════════════

Pattern: Script
- Standalone executable for utility/testing purposes.
"""

import sys
import os
import traceback
import numpy as np

# Add Implementation directory to path (parent of scripts)
current_dir = os.path.dirname(os.path.abspath(__file__))
implementation_dir = os.path.dirname(current_dir)
if implementation_dir not in sys.path:
    sys.path.insert(0, implementation_dir)

from v2_logic.models.embedding_engine import EmbeddingEngine
from v2_logic.models.clustering_engine import ClusteringEngine


def test_integration():
    """
    Test the integration of EmbeddingEngine and ClusteringEngine.
    """
    print("Initializing Engines...")
    try:
        embedder = EmbeddingEngine(model_name="vit_large")
        clusterer = ClusteringEngine(eps=0.5, min_samples=2)
        print("Engines initialized.")
    except (ImportError, RuntimeError, ValueError) as e:
        print(f"Initialization Failed: {e}")
        traceback.print_exc()
        return

    # Create dummy images (3 images similar, 2 different)
    print("Generating dummy images...")
    images = [
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(5)
    ]

    # 1. Embed
    print("Running DINOv2 Embedding...")
    try:
        embeddings = embedder.embed_regions(images)
        print(f"Embeddings shape: {embeddings.shape}")
        if embeddings.shape[1] != 1024:
            print(f"WARNING: Expected 1024 dim (ViT-L), got {embeddings.shape[1]}")
    except RuntimeError as e:
        print(f"Embedding Failed (RuntimeError): {e}")
        traceback.print_exc()
        return
    except ValueError as e:
        print(f"Embedding Failed (ValueError): {e}")
        traceback.print_exc()
        return

    # 2. Cluster
    print("Running DBSCAN Clustering...")
    try:
        labels, clusters = clusterer.fit_predict(embeddings)
        print(f"Labels: {labels}")
        print(f"Number of clusters found: {len(clusters)}")
        for c in clusters:
            print(f"  Cluster {c.cluster_id}: Count={c.count}, Label={c.label}")

    except ValueError as e:
        print(f"Clustering Failed (ValueError): {e}")
        traceback.print_exc()
        return

    print("Test passed!")


if __name__ == "__main__":
    test_integration()
