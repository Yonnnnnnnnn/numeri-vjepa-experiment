"""
Phase 4 Perception Upgrades - Unit Tests

Tests for Union Masking, Multi-Shield Fusion, Graph State updates,
and CountVid Adaptive Sensitivity.

CFG Structure:
===============================================================================
Start Symbol    : TestPhase4Perception (this module)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <TestUnionMasking>       → Mask merge and cluster tests                 │
  │  <TestMultiShield>        → Shield evaluation and confidence tests       │
  │  <TestGraphState>         → State field validation                       │
  │  <TestAdaptiveSensitivity>→ CountVid threshold adjustment tests          │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <SegmentationEngine> ← from models.segmentation_engine                  │
  │  <FusionEngineV2>     ← from models.fusion_engine_v2                     │
  │  <PerceptionState>    ← from types.graph_state                           │
  │  <CountVidEngine>     ← from models.count_vid_engine                     │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : numpy.ndarray, float, int, bool

Production Rules:
  TestPhase4Perception → <TestUnionMasking> + <TestMultiShield>
                       + <TestGraphState> + <TestAdaptiveSensitivity>
===============================================================================

# Simple implementation, no pattern needed
"""

import sys
import os
import unittest

import numpy as np

# Ensure v2_logic is on the path
V2_LOGIC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if V2_LOGIC_PATH not in sys.path:
    sys.path.insert(0, V2_LOGIC_PATH)

IMPL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if IMPL_PATH not in sys.path:
    sys.path.insert(0, IMPL_PATH)

# Import targets under test
from models.segmentation_engine import SegmentResult  # noqa: E402
from models.fusion_engine_v2 import (  # noqa: E402
    FusionEngineV2,
    FusionResult,
    ShieldResult,
)

# Cannot use `from types.graph_state import ...` because Python's built-in
# `types` module shadows our `v2_logic/types` package.
# Load directly from file path instead.
import importlib.util  # noqa: E402

_gs_path = os.path.join(V2_LOGIC_PATH, "types", "graph_state.py")
_gs_spec = importlib.util.spec_from_file_location("graph_state", _gs_path)
_gs_mod = importlib.util.module_from_spec(_gs_spec)
_gs_spec.loader.exec_module(_gs_mod)
PerceptionState = _gs_mod.PerceptionState


class TestUnionMasking(unittest.TestCase):
    """Test Union Masking logic in SegmentationEngine."""

    def setUp(self):
        """Create mock SegmentResult for testing."""
        # Import SegmentationEngine only for methods; model won't be loaded
        from models.segmentation_engine import SegmentationEngine

        self.engine = SegmentationEngine.__new__(SegmentationEngine)

    def test_01_no_merge_needed(self):
        """Non-overlapping masks should not be merged."""
        print("\n[Test] No merge needed (disjoint masks)")

        mask_a = np.zeros((100, 100), dtype=bool)
        mask_b = np.zeros((100, 100), dtype=bool)
        mask_a[10:30, 10:30] = True
        mask_b[60:80, 60:80] = True

        result = SegmentResult(
            masks=[mask_a, mask_b],
            boxes=[(10, 10, 30, 30), (60, 60, 80, 80)],
            scores=[0.9, 0.8],
            num_segments=2,
        )

        merged = self.engine.merge_overlapping_masks(result, iou_threshold=0.5)
        print(f"   Input: {result.num_segments}, Output: {merged.num_segments}")
        self.assertEqual(merged.num_segments, 2, "Disjoint masks should stay separate")

    def test_02_merge_overlapping(self):
        """Highly overlapping masks should be merged."""
        print("\n[Test] Merge overlapping masks")

        mask_a = np.zeros((100, 100), dtype=bool)
        mask_b = np.zeros((100, 100), dtype=bool)
        mask_a[10:50, 10:50] = True  # Large region
        mask_b[15:45, 15:45] = True  # Subset inside A (high IoU)

        result = SegmentResult(
            masks=[mask_a, mask_b],
            boxes=[(10, 10, 50, 50), (15, 15, 45, 45)],
            scores=[0.9, 0.85],
            num_segments=2,
        )

        merged = self.engine.merge_overlapping_masks(result, iou_threshold=0.5)
        print(f"   Input: {result.num_segments}, Output: {merged.num_segments}")
        self.assertEqual(
            merged.num_segments, 1, "Overlapping masks should merge into 1"
        )

    def test_03_cluster_detection_no_depth(self):
        """Without depth map, all masks form one cluster."""
        print("\n[Test] Cluster detection without depth")

        mask_a = np.zeros((100, 100), dtype=bool)
        mask_b = np.zeros((100, 100), dtype=bool)
        mask_a[10:30, 10:30] = True
        mask_b[60:80, 60:80] = True

        result = SegmentResult(
            masks=[mask_a, mask_b],
            boxes=[(10, 10, 30, 30), (60, 60, 80, 80)],
            scores=[0.9, 0.8],
            num_segments=2,
        )

        clusters = self.engine.detect_volumetric_clusters(result, depth_map=None)
        print(f"   Clusters: {len(clusters)}")
        self.assertEqual(len(clusters), 1, "Without depth, all masks = 1 cluster")
        self.assertEqual(len(clusters[0]["mask_indices"]), 2)

    def test_04_cluster_detection_with_depth(self):
        """Masks at different depths should form separate clusters."""
        print("\n[Test] Cluster detection with depth")

        mask_a = np.zeros((100, 100), dtype=bool)
        mask_b = np.zeros((100, 100), dtype=bool)
        mask_a[10:30, 10:30] = True
        mask_b[60:80, 60:80] = True

        # Depth map: mask_a is close (depth=0.3), mask_b is far (depth=0.9)
        depth_map = np.ones((100, 100), dtype=np.float32) * 0.5
        depth_map[10:30, 10:30] = 0.3
        depth_map[60:80, 60:80] = 0.9

        result = SegmentResult(
            masks=[mask_a, mask_b],
            boxes=[(10, 10, 30, 30), (60, 60, 80, 80)],
            scores=[0.9, 0.8],
            num_segments=2,
        )

        clusters = self.engine.detect_volumetric_clusters(
            result, depth_map=depth_map, depth_tolerance=0.15
        )
        print(f"   Clusters: {len(clusters)}")
        self.assertEqual(len(clusters), 2, "Different depths → separate clusters")


class TestMultiShield(unittest.TestCase):
    """Test Multi-Shield Fusion Architecture."""

    def setUp(self):
        self.engine = FusionEngineV2()

    def _make_clean_fusion_result(self):
        """Create a clean (no anomaly) FusionResult."""
        return FusionResult(
            residual_spike_energy=0.05,
            total_spike_energy=1.0,
            residue_ratio=0.05,
            unexplained_blobs=[],
            has_spatial_anomaly=False,
            has_volumetric_anomaly=False,
            motion_compensated=False,
            camera_motion_energy=0.0,
        )

    def test_05_all_shields_pass(self):
        """When all data is clean, fusion confidence should be high."""
        print("\n[Test] All shields pass")

        result = self._make_clean_fusion_result()
        result = self.engine.run_shields(
            fusion_result=result,
            n_visible=5,
            n_volumetric_range=(4, 6),
            track_similarities=[0.9, 0.85, 0.92],
        )

        print(f"   Shield scores: {result.shield_scores}")
        print(f"   Fusion confidence: {result.fusion_confidence:.3f}")

        self.assertGreater(
            result.fusion_confidence, 0.6, "Clean data should pass threshold"
        )
        self.assertIn("spatial", result.shield_scores)
        self.assertIn("volumetric", result.shield_scores)
        self.assertIn("latent", result.shield_scores)

    def test_06_spatial_shield_fails(self):
        """High residue should drop spatial confidence."""
        print("\n[Test] Spatial shield failure")

        result = self._make_clean_fusion_result()
        result.residue_ratio = 0.80  # 80% unexplained energy!
        result.has_spatial_anomaly = True

        result = self.engine.run_shields(
            fusion_result=result,
            n_visible=5,
            n_volumetric_range=(4, 6),
            track_similarities=[0.9, 0.85, 0.92],
        )

        print(f"   Spatial confidence: {result.shield_scores['spatial']:.3f}")
        print(f"   Fusion confidence: {result.fusion_confidence:.3f}")

        self.assertLess(
            result.shield_scores["spatial"],
            0.5,
            "High residue should lower spatial confidence",
        )

    def test_07_volumetric_shield_fails(self):
        """Count outside volume range should drop volumetric confidence."""
        print("\n[Test] Volumetric shield failure")

        result = self._make_clean_fusion_result()
        result = self.engine.run_shields(
            fusion_result=result,
            n_visible=20,  # Way more than expected
            n_volumetric_range=(4, 6),
            track_similarities=[0.9, 0.85, 0.92],
        )

        print(f"   Volumetric confidence: {result.shield_scores['volumetric']:.3f}")
        print(f"   Fusion confidence: {result.fusion_confidence:.3f}")

        self.assertLess(
            result.shield_scores["volumetric"],
            0.5,
            "Count far outside range should lower volumetric confidence",
        )

    def test_08_low_confidence_triggers_warning(self):
        """When all shields fail, confidence should be below threshold."""
        print("\n[Test] All shields fail → low confidence")

        result = self._make_clean_fusion_result()
        result.residue_ratio = 0.90  # Spatial fail
        result.has_spatial_anomaly = True

        result = self.engine.run_shields(
            fusion_result=result,
            n_visible=50,  # Volumetric fail (range 4-6)
            n_volumetric_range=(4, 6),
            track_similarities=[0.1, 0.05],  # Latent fail
        )

        print(f"   Shield scores: {result.shield_scores}")
        print(f"   Fusion confidence: {result.fusion_confidence:.3f}")

        self.assertLess(
            result.fusion_confidence,
            0.6,
            "All shields failing should drop below 0.6 threshold",
        )


class TestGraphState(unittest.TestCase):
    """Test PerceptionState Phase 4 fields."""

    def test_09_shield_fields_exist(self):
        """PerceptionState should have shield_scores and fusion_confidence."""
        print("\n[Test] PerceptionState Phase 4 fields")

        state = PerceptionState()

        self.assertIsInstance(state.shield_scores, dict)
        self.assertEqual(state.fusion_confidence, 1.0)

        # Simulate update
        state.shield_scores = {"spatial": 0.95, "volumetric": 0.8, "latent": 0.6}
        state.fusion_confidence = 0.82

        print(f"   shield_scores: {state.shield_scores}")
        print(f"   fusion_confidence: {state.fusion_confidence}")
        self.assertEqual(len(state.shield_scores), 3)


class TestAdaptiveSensitivity(unittest.TestCase):
    """Test CountVid adaptive threshold adjustment."""

    def test_10_overcounting_raises_threshold(self):
        """Positive feedback (over-counting) should raise threshold."""
        print("\n[Test] Over-counting raises threshold")

        # Create a minimal CountVidEngine without model loading
        from models.count_vid_engine import CountVidEngine, CONF_THRESH

        engine = CountVidEngine.__new__(CountVidEngine)
        engine.confidence_thresh = CONF_THRESH
        engine.model = None

        old = engine.confidence_thresh
        new = engine.update_sensitivity(feedback_score=1.0, density_hint=0.5)

        print(f"   Old threshold: {old:.3f}")
        print(f"   New threshold: {new:.3f}")
        self.assertGreater(new, old, "Over-counting feedback should raise threshold")

    def test_11_undercounting_lowers_threshold(self):
        """Negative feedback (under-counting) should lower threshold."""
        print("\n[Test] Under-counting lowers threshold")

        from models.count_vid_engine import CountVidEngine, CONF_THRESH

        engine = CountVidEngine.__new__(CountVidEngine)
        engine.confidence_thresh = CONF_THRESH
        engine.model = None

        old = engine.confidence_thresh
        new = engine.update_sensitivity(feedback_score=-1.0, density_hint=0.5)

        print(f"   Old threshold: {old:.3f}")
        print(f"   New threshold: {new:.3f}")
        self.assertLess(new, old, "Under-counting feedback should lower threshold")

    def test_12_threshold_clamping(self):
        """Threshold should be clamped to [0.10, 0.60]."""
        print("\n[Test] Threshold clamping")

        from models.count_vid_engine import CountVidEngine

        engine = CountVidEngine.__new__(CountVidEngine)
        engine.model = None

        # Push very low
        engine.confidence_thresh = 0.11
        result_low = engine.update_sensitivity(feedback_score=-1.0)
        print(f"   After strong under-count: {result_low:.3f}")
        self.assertGreaterEqual(result_low, 0.10)

        # Push very high
        engine.confidence_thresh = 0.59
        result_high = engine.update_sensitivity(feedback_score=1.0)
        print(f"   After strong over-count: {result_high:.3f}")
        self.assertLessEqual(result_high, 0.60)


if __name__ == "__main__":
    print("=" * 70)
    print("Phase 4: Perception Upgrades - Test Suite")
    print("=" * 70)

    unittest.main(verbosity=2)
