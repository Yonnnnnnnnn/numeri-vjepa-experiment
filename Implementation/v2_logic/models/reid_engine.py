"""
Re-Identification Engine

Matches detections across frames/loops to support Recursive Intent.
Uses IoU for spatial matching and V-JEPA/CLIP features for visual matching.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : ReIDEngine (this file)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <ReIDEngine>     → Main class for object tracking                        │
  │  <Track>          → Data structure for a single tracked object            │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <MathUtils>      ← from v2_logic.utils.math_utils                        │
  │  <linear_sum_assignment> ← from scipy.optimize (Hungarian Algo)           │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : int, float, List, Dict, np.ndarray

Production Rules:
  ReIDEngine      → __init__ + match_detections + update_tracks
  <Track>         → id + bbox + features + history
═══════════════════════════════════════════════════════════════════════════════

Pattern: Strategy (Matching)
- Can switch between IoU-only or Feature-based matching strategies.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment
from ..utils.math_utils import MathUtils


@dataclass
class Track:
    """Represents a single tracked object."""

    track_id: int
    bbox: Dict[str, float]  # {'x':, 'y':, 'w':, 'h':}
    features: Optional[np.ndarray] = None
    age: int = 0  # Number of frames since first seen
    missed: int = 0  # Number of frames missed (for deletion)
    confidence: float = 0.0


class ReIDEngine:
    """
    Re-Identification Engine for Recursive Intent loops.
    """

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

    def match_detections(
        self, detections: List[Dict], frame_idx: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Match new detections to existing tracks using Hungarian Algorithm.

        Args:
            detections: List of detection dicts [{'bbox':..., 'score':..., 'features':...}]
            frame_idx: Current frame index

        Returns:
            Tuple(matches_list, new_detections_list)
        """
        if not self.tracks:
            # Initialize tracks for all detections
            new_detections = []
            for det in detections:
                new_id = self.next_id
                self.next_id += 1
                new_track = Track(
                    track_id=new_id,
                    bbox=det["bbox"],
                    features=det.get("features"),
                    confidence=det["score"],
                )
                self.tracks.append(new_track)
                new_detections.append({**det, "track_id": new_id})
            return [], new_detections

        if not detections:
            # Mark all tracks as missed
            for track in self.tracks:
                track.missed += 1
            # Cleanup dead tracks
            self.tracks = [t for t in self.tracks if t.missed < self.max_missed]
            return [], []

        # Cost Matrix: Rows = Tracks, Cols = Detections
        num_tracks = len(self.tracks)
        num_dets = len(detections)
        cost_matrix = np.ones((num_tracks, num_dets)) * 100.0  # High cost default

        for t_idx, track in enumerate(self.tracks):
            for d_idx, det in enumerate(detections):
                # 1. IoU Cost (1 - IoU)
                iou = MathUtils.calculate_bbox_overlap(track.bbox, det["bbox"])
                iou_cost = 1.0 - iou

                # 2. Visual Cost (1 - Similarity)
                visual_cost = 1.0
                if track.features is not None and det.get("features") is not None:
                    sim = MathUtils.calculate_vector_similarity(
                        track.features, det["features"]
                    )
                    visual_cost = 1.0 - sim
                elif track.features is None and det.get("features") is None:
                    # Fallback to IoU only if no features available
                    visual_cost = 0.0

                # Combine costs (Weighted average or hard priority)
                # Here we use simple weighted sum if features exist
                if track.features is not None and det.get("features") is not None:
                    final_cost = 0.4 * iou_cost + 0.6 * visual_cost
                else:
                    final_cost = iou_cost

                # Gating
                if (
                    iou < self.iou_threshold
                    and visual_cost > 0.5  # Poor spatial overlap & not visually similar
                ):
                    final_cost = 100.0  # Impossible

                cost_matrix[t_idx, d_idx] = final_cost

        # Hungarian Assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched_detections = []
        unmatched_det_indices = set(range(num_dets))

        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < 1.0:  # Valid match threshold
                # Update track
                track = self.tracks[row]
                det = detections[col]

                track.bbox = det["bbox"]
                track.confidence = det["score"]
                track.missed = 0
                track.age += 1
                if det.get("features") is not None:
                    # EMA update for features could go here, for now simple replace
                    track.features = det["features"]

                matched_detections.append({**det, "track_id": track.track_id})
                unmatched_det_indices.discard(col)
            else:
                # Match rejected by threshold
                self.tracks[row].missed += 1

        # Handle unmatched tracks (already incremented 'missed' in the loop check above? No.)
        # The loop iterates over MATCHES. We need to find unmatched tracks.
        matched_track_indices = set(row_indices)
        for t_idx in range(num_tracks):
            if t_idx not in matched_track_indices:
                self.tracks[t_idx].missed += 1
            # Also check if it was 'matched' but rejected by cost threshold
            elif (
                t_idx in matched_track_indices
                and cost_matrix[t_idx, col_indices[list(row_indices).index(t_idx)]]
                >= 1.0
            ):
                # Already handled in the loop else block
                pass

        # Create new tracks for unmatched detections
        new_detections = []
        for d_idx in unmatched_det_indices:
            det = detections[d_idx]
            new_id = self.next_id
            self.next_id += 1
            new_track = Track(
                track_id=new_id,
                bbox=det["bbox"],
                features=det.get("features"),
                confidence=det["score"],
            )
            self.tracks.append(new_track)
            new_detections.append({**det, "track_id": new_id})

        # Cleanup dead tracks
        self.tracks = [t for t in self.tracks if t.missed < self.max_missed]

        return matched_detections, new_detections
