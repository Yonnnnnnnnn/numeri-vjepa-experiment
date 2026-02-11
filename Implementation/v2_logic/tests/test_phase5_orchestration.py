import os
import sys
import numpy as np
from typing import Dict, Any

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import Implementation.v2_logic.controllers.recursive_flow as flow

# =============================================================================
# MONKEY-PATCHING ENGINES
# =============================================================================


class MockResult(list):
    def __init__(self):
        super().__init__([])
        self.masks = [np.ones((224, 224), dtype=bool)]
        self.num_segments = 1
        self.scores = [0.9]
        self.boxes = [(0, 0, 10, 10)]
        self.stats = {"mean": 1.0}
        self.has_depth = True
        self.depth_map = np.ones((224, 224))
        self.residual_spike_energy = 0.01
        self.unexplained_blobs = []
        self.shield_scores = {"spatial": 0.9, "volumetric": 0.9, "latent": 0.9}
        self.fusion_confidence = 0.9
        self.reasoning_text = "Mock reasoning"
        self.hypothesis = "Match"

    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return self


class MockEngine:
    def __getattr__(self, name):
        if name == "generate_events":
            return lambda *args, **kwargs: np.zeros((10, 4))
        if name == "count":
            return lambda *args, **kwargs: (1, [])
        if name == "predict":
            return lambda *args, **kwargs: np.array([0.85])
        if name == "find_golden_alpha":
            return lambda *args, **kwargs: 0.123
        if name == "analyze_specularity":
            return lambda *args, **kwargs: 0.42
        if name == "match_latent":
            return lambda *args, **kwargs: []
        if name == "match_detections":
            return lambda *args, **kwargs: ([], [])  # Return (matched, new)
        if name == "estimate_depth":
            res = MockResult()
            res.has_depth = True
            res.depth_map = np.ones((224, 224))
            res.stats = {"mean": 1.0}
            return lambda *args, **kwargs: res
        if name == "segment_frame":
            res = MockResult()
            res.masks = [np.ones((224, 224), dtype=bool)]
            res.num_segments = 1
            return lambda *args, **kwargs: res
        if name == "merge_overlapping_masks":
            return lambda *args, **kwargs: (
                args[0] if args else kwargs.get("result", MockResult())
            )
        if name == "detect_volumetric_clusters":
            return lambda *args, **kwargs: [
                {
                    "mask_indices": [0],
                    "mean_depth": 1.0,
                    "combined_mask": np.ones((224, 224), dtype=bool),
                }
            ]
        if name == "fuse_spike_mask":
            res = MockResult()
            res.residual_spike_energy = 0.01
            res.unexplained_blobs = []
            return lambda *args, **kwargs: res
        if name == "run_shields":
            return lambda *args, **kwargs: (
                args[0] if args else kwargs.get("fusion_result", MockResult())
            )
        if name == "evaluate":
            from Implementation.v2_logic.models.logic_gate import (
                GateDecision,
                AnomalyType,
            )

            return lambda *args, **kwargs: GateDecision(
                action="exit",
                anomaly_type=AnomalyType.NONE,
                confidence=0.9,
                rule_applied="Mock",
                reasoning="Mock",
                details={},
            )
        return lambda *args, **kwargs: MockResult()

    def __call__(self, *args, **kwargs):
        return MockResult()


def mock_get_engine():
    return MockEngine()


# Patch all engines
engine_getters = [
    "get_segmentation_engine",
    "get_depth_engine",
    "get_countvid_engine",
    "get_fusion_engine",
    "get_logic_gate",
    "get_slm_engine",
    "get_reid_engine",
    "get_v2e_engine",
    "get_vjepa_engine",
    "get_dinov2_engine",
    "get_density_predictor",
    "get_alphashape_wrapper",
]

for getter in engine_getters:
    setattr(flow, getter, mock_get_engine)

# =============================================================================
# IMPORTS AFTER PATCHING
# =============================================================================

from Implementation.v2_logic.controllers.recursive_flow import create_recursive_flow_app
from Implementation.v2_logic.types.graph_state import (
    GlobalContext,
    PerceptionState,
    DecisionState,
    OutputAccumulator,
)

# =============================================================================
# TESTS
# =============================================================================


def test_graph_compilation():
    print("\n[Test] Compiling Recursive Flow Graph...")
    app = flow.create_recursive_flow_app()
    assert app is not None
    print("Result: Success (App Compiled)")


def test_single_pass_mock():
    print("\n[Test] Running Single Pass (Mock Input)...")
    app = flow.create_recursive_flow_app()

    state = {
        "ctx": GlobalContext(
            session_id="test_session_1", start_time=1738224000.0, main_intent=["ball"]
        ),
        "perception": PerceptionState(
            image=np.zeros((224, 224, 3), dtype=np.uint8),
            v2e_spike_map=np.zeros((224, 224), dtype=np.float32),
            n_visible=1,
            total_observed_volume=0.0012,
        ),
        "decision": DecisionState(),
        "output": OutputAccumulator(),
    }

    result = app.invoke(state)

    print(f"Final Action: {result['decision'].status}")
    print(f"N_volumetric Range: {result['perception'].n_volumetric_range}")
    print(f"Loop Count: {result['decision'].loop_count}")

    assert "n_volumetric_range" in result["perception"].model_dump()
    print("Result: Success (Single Pass Completed)")


if __name__ == "__main__":
    test_graph_compilation()
    try:
        test_single_pass_mock()
        print("\n[ALL TESTS PASSED] Phase 5 Orchestration is valid.")
    except Exception as e:
        print(f"\n[TEST FAILED] {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
