"""
Unit Tests for Logic Gate

Verifies the "Math Guards" decision logic.
"""

import pytest
from v2_logic.models.logic_gate import LogicGate, AnomalyType


def test_high_confidence_pass(logic_gate):
    """Rule 1: High confidence, no anomalies -> EXIT"""
    decision = logic_gate.evaluate(
        n_visible=5,
        n_volumetric_range=(4, 6),
        residue_ratio=0.05,  # Low residue
        unexplained_blob_area=0.02,  # Low unexplained area
        detection_confidence=0.95,  # High confidence
        current_loop_count=0,
    )
    assert decision.action == "exit"
    assert decision.anomaly_type == AnomalyType.NONE
    assert "Rule1" in decision.rule_applied


def test_low_confidence_fail(logic_gate):
    """Rule 2: Low confidence -> EXIT (Noise)"""
    decision = logic_gate.evaluate(
        n_visible=1,
        n_volumetric_range=(0, 0),
        residue_ratio=0.0,
        unexplained_blob_area=0.0,
        detection_confidence=0.2,  # Very low
        current_loop_count=0,
    )
    assert decision.action == "exit"
    assert decision.anomaly_type == AnomalyType.NONE
    assert "Rule2" in decision.rule_applied


def test_spatial_anomaly(logic_gate):
    """Rule 3: High residual energy -> LOOP"""
    decision = logic_gate.evaluate(
        n_visible=5,
        n_volumetric_range=(4, 6),
        residue_ratio=0.30,  # > 0.15 threshold
        unexplained_blob_area=0.0,
        detection_confidence=0.8,
        current_loop_count=0,
        has_spatial_anomaly=True,
    )
    assert decision.action == "loop"
    assert decision.anomaly_type == AnomalyType.SPATIAL
    assert "SpatialAnomaly" in decision.rule_applied


def test_volumetric_anomaly(logic_gate):
    """Rule 3: Count mismatch -> LOOP"""
    decision = logic_gate.evaluate(
        n_visible=10,  # Mismatch!
        n_volumetric_range=(4, 6),
        residue_ratio=0.05,
        unexplained_blob_area=0.0,
        detection_confidence=0.9,
        current_loop_count=0,
        has_volumetric_anomaly=True,
    )
    assert decision.action == "loop"
    assert decision.anomaly_type == AnomalyType.VOLUMETRIC
    assert "VolumetricAnomaly" in decision.rule_applied


def test_max_loop_safety(logic_gate):
    """Safety: Max loops reached -> EXIT"""
    decision = logic_gate.evaluate(
        n_visible=10,
        n_volumetric_range=(4, 6),
        residue_ratio=0.30,
        unexplained_blob_area=0.0,
        detection_confidence=0.9,
        current_loop_count=3,  # Max reached
        has_spatial_anomaly=True,
    )
    assert decision.action == "exit"
    assert "MaxLoopSafety" in decision.rule_applied


def test_check_volumetric_anomaly_helper():
    """Verify helper logic"""
    # Inside range
    assert not LogicGate.check_volumetric_anomaly(5, (4, 6))
    # Below
    assert LogicGate.check_volumetric_anomaly(3, (4, 6))
    # Above
    assert LogicGate.check_volumetric_anomaly(7, (4, 6))
    # No range (0,0)
    assert not LogicGate.check_volumetric_anomaly(100, (0, 0))
