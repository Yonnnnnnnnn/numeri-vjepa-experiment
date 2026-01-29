"""
Pytest Fixtures for Recursive Intent System

Provides mock states, contexts, and engine stubs for unit testing.
"""

import pytest
import numpy as np
from typing import Dict, Any, List

# Add parent directory to path to import v2_logic
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from v2_logic.types.graph_state import (
    RecursiveFlowState,
    GlobalContext,
    PerceptionState,
    DecisionState,
    OutputAccumulator,
    create_initial_state,
)
from v2_logic.models.logic_gate import LogicGate, AnomalyType


@pytest.fixture
def mock_context() -> GlobalContext:
    return GlobalContext(
        session_id="test_session",
        main_intent=["test_object"],
        start_time=1000.0,
        unit_volume_prior=0.001,
        depth_scale_factor=10.0,
    )


@pytest.fixture
def mock_perception() -> PerceptionState:
    return PerceptionState(
        current_frame_idx=0,
        n_visible=5,
        n_volumetric_range=(4, 6),
        spike_energy=100.0,
        residual_spike_energy=5.0,  # 5% residue
        masks=[np.zeros((10, 10))],  # Dummy mask
        raw_detections=[
            {
                "bbox": {"x": 10, "y": 10, "w": 50, "h": 50},
                "score": 0.9,
                "label": "test_object",
            }
        ],
    )


@pytest.fixture
def mock_decision() -> DecisionState:
    return DecisionState(status="processing", loop_count=0)


@pytest.fixture
def mock_state(
    mock_context: GlobalContext,
    mock_perception: PerceptionState,
    mock_decision: DecisionState,
) -> RecursiveFlowState:
    return RecursiveFlowState(
        ctx=mock_context,
        perception=mock_perception,
        decision=mock_decision,
        output=OutputAccumulator(),
    )


@pytest.fixture
def logic_gate() -> LogicGate:
    return LogicGate(
        high_confidence_threshold=0.85,
        low_confidence_threshold=0.4,
        unexplained_area_threshold=0.10,
        residue_threshold=0.15,
    )
