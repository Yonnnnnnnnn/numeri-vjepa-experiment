"""
Test Latent Locking & Cosine Similarity for V-JEPA

Tests that V-JEPA's `extract_object_latent` produces reliable vectors
that can be matched using Cosine Similarity.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : TestLatentLocking (this module)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <TestFunction>  → test_latent_cosine_similarity                          │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <VJEPAEngine>   ← from v2_logic.models.v_jepa_engine (Latent Engine)     │
  │  <pytest>        ← from pytest (Testing framework)                        │
  │  <torch>         ← from torch (Tensor operations)                         │
  │  <numpy>         ← from numpy (Math operations)                           │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : str, float, np.ndarray, torch.Tensor

Production Rules:
  TestLatentLocking  → imports + <TestFunction>
═══════════════════════════════════════════════════════════════════════════════
"""

import pytest
import torch
import numpy as np

from v2_logic.models.v_jepa_engine import VJEPAEngine


def test_latent_cosine_similarity():
    vjepa = VJEPAEngine(device="cpu")  # Use CPU for quick test

    # 1. Create a dummy image tensor (1, 3, 224, 224)
    dummy_image = torch.ones((1, 3, 224, 224))

    # Add a specific "hotspot" (feature) in the center
    # This ensures the latent representation isn't perfectly uniform
    dummy_image[:, :, 100:124, 100:124] = 0.5

    # 2. Encode
    vjepa.encode(dummy_image)

    # 3. Extract latent anchor for the hotspot area
    # Bbox: [x, y, w, h] normalized
    hotspot_bbox = {"x": 100 / 224, "y": 100 / 224, "w": 24 / 224, "h": 24 / 224}
    anchor_latent = vjepa.extract_object_latent(hotspot_bbox)

    if hasattr(anchor_latent, "detach"):
        anchor_latent = anchor_latent.detach().cpu().numpy()

    assert anchor_latent is not None
    assert anchor_latent.shape[0] > 0

    # 4. Extract latent for EXACT SAME bbox
    same_bbox_latent = vjepa.extract_object_latent(hotspot_bbox)
    if hasattr(same_bbox_latent, "detach"):
        same_bbox_latent = same_bbox_latent.detach().cpu().numpy()

    # Calculate Cosine Similarity
    sim_exact = np.dot(anchor_latent.flatten(), same_bbox_latent.flatten()) / (
        np.linalg.norm(anchor_latent) * np.linalg.norm(same_bbox_latent) + 1e-9
    )

    # Exact match should be 1.0 (or very close)
    assert sim_exact > 0.99, f"Exact match similarity was {sim_exact}, expected > 0.99"
    print(f"Exact match similarity: {sim_exact:.4f}")

    # 5. Extract latent for slightly shifted bbox (to test overlap)
    shifted_bbox = {"x": 90 / 224, "y": 90 / 224, "w": 24 / 224, "h": 24 / 224}
    shifted_latent = vjepa.extract_object_latent(shifted_bbox)
    if hasattr(shifted_latent, "detach"):
        shifted_latent = shifted_latent.detach().cpu().numpy()

    sim_shifted = np.dot(anchor_latent.flatten(), shifted_latent.flatten()) / (
        np.linalg.norm(anchor_latent) * np.linalg.norm(shifted_latent) + 1e-9
    )

    print(f"Shifted match similarity: {sim_shifted:.4f}")

    # 6. Extract latent for totally different bbox (empty area)
    empty_bbox = {"x": 10 / 224, "y": 10 / 224, "w": 24 / 224, "h": 24 / 224}
    empty_latent = vjepa.extract_object_latent(empty_bbox)
    if hasattr(empty_latent, "detach"):
        empty_latent = empty_latent.detach().cpu().numpy()

    sim_empty = np.dot(anchor_latent.flatten(), empty_latent.flatten()) / (
        np.linalg.norm(anchor_latent) * np.linalg.norm(empty_latent) + 1e-9
    )

    print(f"Empty area match similarity: {sim_empty:.4f}")

    # Check that empty match is lower than exact match
    assert sim_empty < 0.90, f"Empty area similarity {sim_empty} shouldn't be too high"

    # And check our threshold (>0.85) works to distinguish somewhat
    assert sim_shifted > sim_empty


if __name__ == "__main__":
    test_latent_cosine_similarity()
    print("Latent Distance tests passed successfully!")
