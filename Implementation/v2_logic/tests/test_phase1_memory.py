"""
Phase 1 Unit Tests: Spatio-Temporal Foundation

Tests for PersistentLatentContext (V-JEPA Memory) and
ReID Engine latent-pose matching.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : test_phase1_memory (this file)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <TestPersistentLatentContext> → Buffer mechanics unit tests              │
  │  <TestReIDLatentPose>         → Latent-pose matching tests               │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <PersistentLatentContext> ← from v2_logic.models.v_jepa_engine          │
  │  <ReIDEngine>              ← from v2_logic.models.reid_engine            │
  │  <Track>                   ← from v2_logic.models.reid_engine            │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : int, float, torch.Tensor, np.ndarray

Production Rules:
  test_phase1_memory → <TestPersistentLatentContext> + <TestReIDLatentPose>
═══════════════════════════════════════════════════════════════════════════════

# Simple implementation, no pattern needed
"""

import sys
import os
import unittest

import numpy as np
import torch

# Ensure the project root is on the path
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)

from v2_logic.models.v_jepa_engine import PersistentLatentContext  # noqa: E402
from v2_logic.models.reid_engine import ReIDEngine, Track  # noqa: E402


class TestPersistentLatentContext(unittest.TestCase):
    """Tests for the sliding window frame buffer."""

    def setUp(self):
        self.ctx = PersistentLatentContext(queue_size=4)  # Small for testing

    def test_empty_buffer_raises(self):
        """get_context_tensor should raise if buffer is empty."""
        with self.assertRaises(RuntimeError):
            self.ctx.get_context_tensor()

    def test_single_frame_padding(self):
        """Single frame should be padded to fill the queue."""
        frame = torch.randn(3, 224, 224)
        self.ctx.update(frame)

        context = self.ctx.get_context_tensor()
        self.assertEqual(context.shape, (1, 3, 4, 224, 224))
        # All temporal slices should be identical
        for t in range(4):
            self.assertTrue(torch.equal(context[0, :, t, :, :], frame))

    def test_full_buffer(self):
        """Full buffer should maintain queue_size frames."""
        frames = [torch.randn(3, 224, 224) for _ in range(4)]
        for f in frames:
            self.ctx.update(f)

        self.assertTrue(self.ctx.is_full)
        self.assertEqual(self.ctx.frame_count, 4)

        context = self.ctx.get_context_tensor()
        self.assertEqual(context.shape, (1, 3, 4, 224, 224))

    def test_overflow_discards_oldest(self):
        """Adding more than queue_size frames should drop the oldest."""
        frames = [torch.randn(3, 224, 224) for _ in range(6)]
        for f in frames:
            self.ctx.update(f)

        self.assertEqual(self.ctx.frame_count, 4)  # Still maxlen=4

        context = self.ctx.get_context_tensor()
        # Last frame should be frames[5]
        self.assertTrue(torch.equal(context[0, :, 3, :, :], frames[5]))
        # First frame should be frames[2] (oldest surviving)
        self.assertTrue(torch.equal(context[0, :, 0, :, :], frames[2]))

    def test_batch_dim_squeeze(self):
        """Frame with batch dim (1, C, H, W) should be squeezed."""
        frame = torch.randn(1, 3, 224, 224)
        self.ctx.update(frame)
        self.assertEqual(self.ctx.frame_count, 1)

    def test_invalid_dim_raises(self):
        """2D or 5D tensor should raise ValueError."""
        with self.assertRaises(ValueError):
            self.ctx.update(torch.randn(224, 224))

    def test_reset(self):
        """Reset should clear the buffer."""
        self.ctx.update(torch.randn(3, 224, 224))
        self.assertTrue(self.ctx.is_ready)
        self.ctx.reset()
        self.assertFalse(self.ctx.is_ready)
        self.assertEqual(self.ctx.frame_count, 0)


class TestReIDLatentPose(unittest.TestCase):
    """Tests for V3.1 latent-pose matching in ReIDEngine."""

    def setUp(self):
        self.engine = ReIDEngine(
            iou_threshold=0.3,
            feature_threshold=0.7,
            max_missed=3,
        )

    def _make_det(self, x, y, w, h, score=0.9, latent_pose=None):
        """Helper to create a detection dict."""
        return {
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "score": score,
            "latent_pose": latent_pose,
        }

    def test_first_frame_creates_tracks(self):
        """First call should create new tracks for all detections."""
        latent = np.random.randn(1024).astype(np.float32)
        latent /= np.linalg.norm(latent)

        dets = [self._make_det(0.1, 0.1, 0.2, 0.2, latent_pose=latent)]
        matched, new = self.engine.match_detections(dets, frame_idx=0)

        self.assertEqual(len(matched), 0)
        self.assertEqual(len(new), 1)
        self.assertEqual(new[0]["track_id"], 0)

    def test_identical_latent_matches(self):
        """Same latent-pose should match across frames."""
        latent = np.random.randn(1024).astype(np.float32)
        latent /= np.linalg.norm(latent)

        # Frame 0: initialize
        dets_0 = [self._make_det(0.1, 0.1, 0.2, 0.2, latent_pose=latent)]
        self.engine.match_detections(dets_0, frame_idx=0)

        # Frame 1: same latent, slightly shifted bbox
        dets_1 = [self._make_det(0.12, 0.12, 0.2, 0.2, latent_pose=latent)]
        matched, new = self.engine.match_detections(dets_1, frame_idx=1)

        self.assertEqual(len(matched), 1)
        self.assertEqual(len(new), 0)
        self.assertEqual(matched[0]["track_id"], 0)

    def test_different_latent_creates_new_track(self):
        """Very different latent-pose should create a new track."""
        latent_a = np.random.randn(1024).astype(np.float32)
        latent_a /= np.linalg.norm(latent_a)

        latent_b = np.random.randn(1024).astype(np.float32)
        latent_b /= np.linalg.norm(latent_b)

        # Frame 0: object A
        dets_0 = [self._make_det(0.1, 0.1, 0.2, 0.2, latent_pose=latent_a)]
        self.engine.match_detections(dets_0, frame_idx=0)

        # Frame 1: object B at very different location with different latent
        dets_1 = [self._make_det(0.7, 0.7, 0.2, 0.2, latent_pose=latent_b)]
        matched, new = self.engine.match_detections(dets_1, frame_idx=1)

        # Should create new track (not match to A)
        self.assertEqual(len(new), 1)
        self.assertNotEqual(new[0]["track_id"], 0)

    def test_motion_gate_overrides_iou(self):
        """High latent similarity should override poor IoU (occlusion)."""
        latent = np.random.randn(1024).astype(np.float32)
        latent /= np.linalg.norm(latent)

        # Frame 0: object at top-left
        dets_0 = [self._make_det(0.0, 0.0, 0.1, 0.1, latent_pose=latent)]
        self.engine.match_detections(dets_0, frame_idx=0)

        # Frame 1: same object jumps to bottom-right (IoU ≈ 0)
        # but latent is identical -> Motion Gate should fire
        dets_1 = [self._make_det(0.8, 0.8, 0.1, 0.1, latent_pose=latent)]
        matched, new = self.engine.match_detections(dets_1, frame_idx=1)

        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["track_id"], 0)

    def test_missed_tracks_expire(self):
        """Tracks not matched for max_missed frames should be removed."""
        latent = np.random.randn(1024).astype(np.float32)
        latent /= np.linalg.norm(latent)

        # Frame 0: create track
        dets_0 = [self._make_det(0.1, 0.1, 0.2, 0.2, latent_pose=latent)]
        self.engine.match_detections(dets_0, frame_idx=0)
        self.assertEqual(len(self.engine.tracks), 1)

        # Frames 1-3: no detections (max_missed=3)
        for i in range(1, 4):
            self.engine.match_detections([], frame_idx=i)

        self.assertEqual(len(self.engine.tracks), 0)

    def test_latent_similarity_static(self):
        """Verify cosine similarity computation."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(ReIDEngine._compute_latent_similarity(a, b), 1.0)

        c = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(ReIDEngine._compute_latent_similarity(a, c), 0.0)

    def test_none_latent_returns_zero(self):
        """None latent should return 0 similarity."""
        a = np.array([1.0, 0.0])
        self.assertEqual(ReIDEngine._compute_latent_similarity(a, None), 0.0)
        self.assertEqual(ReIDEngine._compute_latent_similarity(None, None), 0.0)


if __name__ == "__main__":
    unittest.main()
