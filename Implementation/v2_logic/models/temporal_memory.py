"""
Temporal Memory Module

Uses V-JEPA features to track object persistence across video frames.
Detects occluded items by maintaining peak count and feature consistency.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : TemporalMemory (this file)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <TemporalMemory>    → Main class for temporal tracking                   │
  │  <FeatureBuffer>     → Rolling buffer of V-JEPA features                  │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <VisionTransformer>  ← from vjepa_src.models (V-JEPA encoder)            │
  │  <torch.Tensor>       ← from torch (feature tensors)                      │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : int, float, list, tuple

Production Rules:
  TemporalMemory → __init__ + extract_features + update + get_temporal_context
═══════════════════════════════════════════════════════════════════════════════

Pattern: Observer
- Observes video frames over time
- Tracks state changes (object count, feature similarity)
- Notifies fusion module of occluded items
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class TemporalContext:
    """Output of temporal memory analysis."""

    peak_count: int  # Highest count ever observed
    current_visible: int  # Current VLM-reported count
    occluded_estimate: int  # Estimated occluded items
    confidence: float  # Confidence in occlusion estimate (0-1)
    feature_similarity: float  # Similarity to previous frame (0-1)


def load_vjepa_model(device):
    """Load V-JEPA encoder lazily."""
    try:
        from vjepa_src.models.vision_transformer import vit_huge

        print("[TemporalMemory] Loading V-JEPA ViT-Huge...")
        model = vit_huge(img_size=224, patch_size=14, num_frames=1)
        model = model.half().to(device)
        model.eval()
        print("[TemporalMemory] V-JEPA loaded successfully.")
        return model
    except ImportError as e:
        print(f"[TemporalMemory] Warning: Could not load V-JEPA: {e}")
        return None


def extract_frame_features(model, frame_tensor: torch.Tensor) -> torch.Tensor:
    """Extract features from a frame using loaded model."""
    if model is None:
        # Fallback: Use simple mean pooling
        return frame_tensor.mean(dim=(2, 3))

    # Resize to 224x224 for V-JEPA
    resized = F.interpolate(
        frame_tensor, size=(224, 224), mode="bilinear", align_corners=False
    )

    with torch.no_grad():
        features = model(resized.half())  # (B, N, D)
        # Global average pooling over patches
        pooled = features.mean(dim=1)  # (B, D)

    return pooled


class TemporalMemory(nn.Module):
    """
    V-JEPA based temporal memory for tracking object persistence.

    Uses V-JEPA's rich temporal features to:
    1. Detect when objects disappear from view (occlusion)
    2. Validate if objects are truly occluded vs removed
    3. Maintain a "peak count" of maximum items seen
    """

    def __init__(
        self,
        buffer_size: int = 30,
        decay_rate: float = 0.98,
        similarity_threshold: float = 0.7,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.decay_rate = decay_rate
        self.similarity_threshold = similarity_threshold

        # State
        self.feature_buffer = deque(maxlen=buffer_size)
        self.count_history = deque(maxlen=buffer_size)
        self.peak_count = 0
        self.decayed_peak = 0.0

    def compute_similarity(
        self, current_features: torch.Tensor, previous_features: torch.Tensor
    ) -> float:
        """Compute cosine similarity between feature sets."""
        if current_features is None or previous_features is None:
            return 1.0

        # Normalize
        current_norm = F.normalize(current_features.float(), dim=-1)
        previous_norm = F.normalize(previous_features.float(), dim=-1)

        # Cosine similarity
        similarity = (current_norm * previous_norm).sum(dim=-1).mean().item()
        return max(0.0, min(1.0, similarity))

    def update(self, features: torch.Tensor, vlm_count: int) -> TemporalContext:
        """
        Update temporal memory with pre-computed features and VLM count.

        Args:
            features: Pre-computed V-JEPA features (B, D)
            vlm_count: Count reported by VLM/Clustering

        Returns:
            TemporalContext with occlusion analysis
        """
        # Save features
        current_features = features

        # Get previous features
        previous_features = (
            self.feature_buffer[-1] if len(self.feature_buffer) > 0 else None
        )

        # Compute feature similarity
        similarity = self.compute_similarity(current_features, previous_features)

        # Update peak count
        if vlm_count > self.peak_count:
            self.peak_count = vlm_count
            self.decayed_peak = float(vlm_count)

        # Apply decay to peak (handles items actually leaving)
        # Only decay if similarity is low (scene changed significantly)
        if similarity < self.similarity_threshold:
            # Scene changed a lot - items might have been removed
            self.decayed_peak = self.decayed_peak * self.decay_rate

        # Calculate occluded estimate
        # If similarity is high, trust the peak count
        # If similarity is low, trust the decayed peak
        effective_peak = (
            self.peak_count
            if similarity >= self.similarity_threshold
            else int(self.decayed_peak)
        )
        occluded = max(0, effective_peak - vlm_count)

        # Confidence based on how recently we saw the peak
        frames_since_peak = sum(1 for c in self.count_history if c < self.peak_count)
        confidence = max(0.3, 1.0 - (frames_since_peak / self.buffer_size))

        # Store in buffer
        self.feature_buffer.append(current_features.detach())
        self.count_history.append(vlm_count)

        return TemporalContext(
            peak_count=self.peak_count,
            current_visible=vlm_count,
            occluded_estimate=occluded,
            confidence=confidence,
            feature_similarity=similarity,
        )

    def reset(self):
        """Reset temporal memory state."""
        self.feature_buffer.clear()
        self.count_history.clear()
        self.peak_count = 0
        self.decayed_peak = 0.0
