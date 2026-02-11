"""
Re-Identification Engine (V3.1 Latent-Pose)

Matches detections across frames/loops to support Recursive Intent.
Uses IoU for spatial matching and V-JEPA latent-pose features for
identity persistence through occlusion.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : ReIDEngine (this file)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <ReIDEngine>     → Main class for latent-pose tracking                  │
  │  <Track>          → Data structure for a single tracked object           │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <MathUtils>             ← from v2_logic.utils.math_utils                │
  │  <linear_sum_assignment> ← from scipy.optimize (Hungarian Algo)          │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : int, float, List, Dict, np.ndarray

Production Rules:
  ReIDEngine      → __init__ + match_detections + _compute_latent_similarity
  <Track>         → id + bbox + features + latent_pose + golden_alpha + history

Pattern: Strategy (Matching)
- Switches between IoU-only or Latent-Pose matching depending on feature
  availability.
- V3.1 upgrade: latent_pose from V-JEPA replaces generic CLIP features
  as the primary identity signal.
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# pylint: disable=relative-beyond-top-level
from ..utils.math_utils import MathUtils

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """
    Represents a single tracked object with latent-pose identity.

    Attributes:
        track_id: Unique identifier for this track.
        bbox: Bounding box {'x':, 'y':, 'w':, 'h':} in normalized coords.
        features: Legacy visual features (DINOv2/CLIP). Optional fallback.
        latent_pose: V-JEPA latent feature vector for identity matching.
        golden_alpha: Physical signature from AlphaShape calibration.
        age: Number of frames since first seen.
        missed: Consecutive frames where this track was not matched.
        confidence: Detection confidence score.
    """

    track_id: int
    bbox: Dict[str, float]
    features: Optional[np.ndarray] = None
    latent_pose: Optional[np.ndarray] = None
    golden_alpha: Optional[float] = None
    age: int = 0
    missed: int = 0
    confidence: float = 0.0


class ReIDEngine:
    """
    Re-Identification Engine for Recursive Intent loops (V3.1).

    Pattern: Strategy (Matching)
    - Primary: V-JEPA Latent-Pose similarity (identity in latent space).
    - Fallback: IoU + legacy visual features.
    - Motion Gate: If latent similarity is very high (>0.85), IoU is
      ignored, enabling re-identification through occlusion.
    """

    # Cost weights
    IOU_WEIGHT = 0.4
    LATENT_WEIGHT = 0.6
    MOTION_GATE_THRESHOLD = 0.85  # If latent sim > this, skip IoU check

    def __init__(
        self,
        iou_threshold: float = 0.3,
        feature_threshold: float = 0.7,
        max_missed: int = 5,
    ):
        self.iou_threshold = iou_threshold
        self.feature_threshold = feature_threshold
        self.max_missed = max_missed
        self.next_id = 0
        self.tracks: List[Track] = []

    @staticmethod
    def _compute_latent_similarity(
        latent_a: Optional[np.ndarray], latent_b: Optional[np.ndarray]
    ) -> float:
        """
        Compute cosine similarity between two latent-pose vectors.

        Args:
            latent_a: First latent vector (already L2-normalized).
            latent_b: Second latent vector (already L2-normalized).

        Returns:
            Cosine similarity in [0, 1], or 0.0 if either is None.
        """
        if latent_a is None or latent_b is None:
            return 0.0

        dot = float(np.dot(latent_a, latent_b))
        # Clamp to valid range (numerical stability)
        return max(0.0, min(1.0, dot))

    def _create_track(self, det: dict) -> Track:
        """Create a new Track from a detection dict."""
        new_id = self.next_id
        self.next_id += 1
        return Track(
            track_id=new_id,
            bbox=det["bbox"],
            features=det.get("features"),
            latent_pose=det.get("latent_pose"),
            golden_alpha=det.get("golden_alpha"),
            confidence=det.get("score", 0.0),
        )

    def match_detections(
        self, detections: List[Dict], frame_idx: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Match new detections to existing tracks using Hungarian Algorithm
        with V-JEPA latent-pose similarity as the primary cost signal.

        Args:
            detections: List of detection dicts:
                [{'bbox':..., 'score':..., 'features':..., 'latent_pose':...}]
            frame_idx: Current frame index (for logging).

        Returns:
            Tuple(matched_detections, new_detections)
            Each detection dict in the output has 'track_id' appended.
        """
        if not self.tracks:
            new_detections = []
            for det in detections:
                track = self._create_track(det)
                self.tracks.append(track)
                new_detections.append({**det, "track_id": track.track_id})
            logger.debug(
                "[ReID] Frame %d: Initialized %d tracks", frame_idx, len(new_detections)
            )
            return [], new_detections

        if not detections:
            for track in self.tracks:
                track.missed += 1
            self.tracks = [t for t in self.tracks if t.missed < self.max_missed]
            return [], []

        # Build Cost Matrix: Rows = Tracks, Cols = Detections
        num_tracks = len(self.tracks)
        num_dets = len(detections)
        cost_matrix = np.ones((num_tracks, num_dets)) * 100.0

        for t_idx, track in enumerate(self.tracks):
            for d_idx, det in enumerate(detections):
                # 1. IoU Cost
                iou = MathUtils.calculate_bbox_overlap(track.bbox, det["bbox"])
                iou_cost = 1.0 - iou

                # 2. Latent-Pose Cost (V3.1 primary signal)
                latent_sim = self._compute_latent_similarity(
                    track.latent_pose, det.get("latent_pose")
                )
                latent_cost = 1.0 - latent_sim

                # 3. Legacy Visual Cost (fallback if no latent)
                visual_cost = 1.0
                has_latent = (
                    track.latent_pose is not None and det.get("latent_pose") is not None
                )
                has_visual = (
                    track.features is not None and det.get("features") is not None
                )

                if has_latent:
                    # V3.1: Latent-Pose is primary identity signal
                    final_cost = (
                        self.IOU_WEIGHT * iou_cost + self.LATENT_WEIGHT * latent_cost
                    )

                    # Motion Gate: high latent similarity overrides IoU
                    if latent_sim > self.MOTION_GATE_THRESHOLD:
                        final_cost = latent_cost * 0.5
                elif has_visual:
                    # Fallback: Legacy visual matching
                    visual_sim = MathUtils.calculate_vector_similarity(
                        track.features, det["features"]
                    )
                    visual_cost = 1.0 - visual_sim
                    final_cost = (
                        self.IOU_WEIGHT * iou_cost + self.LATENT_WEIGHT * visual_cost
                    )
                else:
                    # IoU only
                    final_cost = iou_cost

                # Gating: reject impossible matches
                if not has_latent and iou < self.iou_threshold and visual_cost > 0.5:
                    final_cost = 100.0

                cost_matrix[t_idx, d_idx] = final_cost

        # Hungarian Assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched_detections = []
        unmatched_det_indices = set(range(num_dets))
        matched_track_indices = set()

        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < 1.0:
                # Valid match
                track = self.tracks[row]
                det = detections[col]

                track.bbox = det["bbox"]
                track.confidence = det.get("score", 0.0)
                track.missed = 0
                track.age += 1

                # Update latent pose (simple replacement; EMA is future work)
                if det.get("latent_pose") is not None:
                    track.latent_pose = det["latent_pose"]

                # Update legacy features
                if det.get("features") is not None:
                    track.features = det["features"]

                # Update golden_alpha if provided
                if det.get("golden_alpha") is not None:
                    track.golden_alpha = det["golden_alpha"]

                matched_detections.append({**det, "track_id": track.track_id})
                unmatched_det_indices.discard(col)
                matched_track_indices.add(row)
            else:
                self.tracks[row].missed += 1
                matched_track_indices.add(row)

        # Increment missed for unmatched tracks
        for t_idx in range(num_tracks):
            if t_idx not in matched_track_indices:
                self.tracks[t_idx].missed += 1

        # Create new tracks for unmatched detections
        new_detections = []
        for d_idx in unmatched_det_indices:
            det = detections[d_idx]
            track = self._create_track(det)
            self.tracks.append(track)
            new_detections.append({**det, "track_id": track.track_id})

        # Cleanup dead tracks
        self.tracks = [t for t in self.tracks if t.missed < self.max_missed]

        logger.debug(
            "[ReID] Frame %d: %d matched, %d new, %d active tracks",
            frame_idx,
            len(matched_detections),
            len(new_detections),
            len(self.tracks),
        )

        return matched_detections, new_detections
