"""
Unit Tests for Re-Identification Engine

Verifies object tracking persistence across frames.
"""

import pytest
import numpy as np
from v2_logic.models.reid_engine import ReIDEngine


@pytest.fixture
def reid_engine():
    # Use max_missed=2 as before
    return ReIDEngine(iou_threshold=0.3, feature_threshold=0.7, max_missed=2)


def test_initial_match(reid_engine):
    """First frame should create new tracks."""
    detections = [
        {"bbox": {"x": 10, "y": 10, "w": 50, "h": 50}, "score": 0.9},
        {"bbox": {"x": 100, "y": 100, "w": 50, "h": 50}, "score": 0.8},
    ]
    matched, new_dets = reid_engine.match_detections(detections, frame_idx=0)

    assert len(matched) == 0
    assert len(new_dets) == 2
    # Logic note: new tracks are created internally AFTER match_detections returns for Phase 3 logic
    # Wait, reid_engine updates `self.tracks` internally?
    # Let's check reid_engine.py.
    # Yes, typically it should updates `self.tracks` with new_dets.
    # If not, we assert on return values.

    # Actually, in reid_engine.py, new tracks are added to self.tracks inside match_detections?
    # Let's assume yes. If it fails, checks reid_engine implementation.
    assert len(reid_engine.tracks) == 2
    assert reid_engine.tracks[0].track_id == 0


def test_iou_match(reid_engine):
    """Subsequent frame with overlap should match."""
    # Frame 0
    det0 = [{"bbox": {"x": 10, "y": 10, "w": 50, "h": 50}, "score": 0.9}]
    reid_engine.match_detections(det0, 0)

    track_id = reid_engine.tracks[0].track_id

    # Frame 1 (Slightly moved)
    det1 = [{"bbox": {"x": 12, "y": 12, "w": 50, "h": 50}, "score": 0.9}]
    matched, new_dets = reid_engine.match_detections(det1, 1)

    assert len(matched) == 1
    assert len(new_dets) == 0
    assert matched[0]["track_id"] == track_id


def test_missed_track(reid_engine):
    """Track should persist if missed, then expire."""
    # Frame 0
    det0 = [{"bbox": {"x": 10, "y": 10, "w": 50, "h": 50}, "score": 0.9}]
    reid_engine.match_detections(det0, 0)

    # Frame 1 (Empty)
    reid_engine.match_detections([], 1)
    assert len(reid_engine.tracks) == 1
    assert reid_engine.tracks[0].missed == 1

    # Frame 2 (Empty)
    # Logic: if missed (will be 2) < max_missed (2) -> False. Removed.
    reid_engine.match_detections([], 2)
    assert len(reid_engine.tracks) == 0


def test_visual_match(reid_engine):
    """Feature similarity should override poor IoU."""
    feat1 = np.array([1.0, 0.0])
    feat2 = np.array([0.95, 0.05])  # Highly similar

    # Frame 0
    det0 = [
        {"bbox": {"x": 10, "y": 10, "w": 50, "h": 50}, "score": 0.9, "features": feat1}
    ]
    reid_engine.match_detections(det0, 0)

    # Frame 1 (Far away but visually identical)
    # IoU = 0
    det1 = [
        {
            "bbox": {"x": 500, "y": 500, "w": 50, "h": 50},
            "score": 0.9,
            "features": feat2,
        }
    ]

    matched, new_dets = reid_engine.match_detections(det1, 1)

    assert len(matched) == 1
    assert len(new_dets) == 0
    assert matched[0]["track_id"] == reid_engine.tracks[0].track_id
