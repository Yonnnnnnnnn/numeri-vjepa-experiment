"""
Logic Gate (Math Guards v1)

Primary decision-making component for the Recursive Intent system.
Implements hierarchical anomaly checking and routing decisions.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : LogicGate (this module)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <LogicGate>       → Main decision gate                                   │
  │  <GateDecision>    → Output dataclass with decision and reasoning         │
  │  <AnomalyType>     → Enum for anomaly types                               │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <FusionResult>     ← from fusion_engine_v2                               │
  │  <PerceptionState>  ← from types.graph_state                              │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : "exit", "loop", float, int, str

Production Rules:
  LogicGate     → __init__ + evaluate
  evaluate      → check_rules → GateDecision
  check_rules   → Rule1(Pass) | Rule2(Fail) | Rule3(Ambiguous)
═══════════════════════════════════════════════════════════════════════════════

Pattern: Chain of Responsibility
- Checks anomalies in hierarchical order: Spatial → Volumetric → Confidence.
- First matching rule determines the action.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Literal, Optional, Tuple

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies detected by Logic Gate."""

    NONE = "none"
    SPATIAL = "spatial"  # High residual spike (something moving outside masks)
    VOLUMETRIC = "volumetric"  # Count doesn't match volume range
    PHYSICS = "physics"  # Count exceeds physical bounds


@dataclass
class GateDecision:
    """Result of Logic Gate evaluation."""

    action: Literal["exit", "loop"]  # Final decision
    anomaly_type: AnomalyType  # Type of anomaly detected (if any)
    confidence: float  # Decision confidence (0-1)
    rule_applied: str  # Which rule triggered the decision
    reasoning: str  # Human-readable explanation
    details: Dict[str, Any]  # Additional details for debugging


class LogicGate:
    """
    Primary decision gate implementing the "Math Guards".

    Rules (evaluated in order):
    1. Rule 1 (PASS): Confidence > 0.85 AND Unexplained_Blob_Area < 10%
       → EXIT (Accept Count)

    2. Rule 2 (FAIL): Confidence < 0.4
       → EXIT (Ignore as noise, accept current count)

    3. Rule 3 (AMBIGUOUS): Confidence between 0.4-0.85 OR anomaly detected
       → LOOP (Trigger SLM for reasoning)

    Pattern: Chain of Responsibility
    """

    def __init__(
        self,
        high_confidence_threshold: float = 0.85,
        low_confidence_threshold: float = 0.4,
        unexplained_area_threshold: float = 0.10,
        residue_threshold: float = 0.15,
        max_loop_count: int = 3,
    ):
        """
        Args:
            high_confidence_threshold: Threshold for confident PASS.
            low_confidence_threshold: Threshold for confident FAIL (noise).
            unexplained_area_threshold: Max unexplained blob area ratio.
            residue_threshold: Max residue ratio before triggering loop.
            max_loop_count: Maximum recursive loops before forced exit.
        """
        self.high_conf = high_confidence_threshold
        self.low_conf = low_confidence_threshold
        self.unexplained_threshold = unexplained_area_threshold
        self.residue_threshold = residue_threshold
        self.residue_threshold = residue_threshold
        self.max_loop_count = max_loop_count

    @staticmethod
    def check_volumetric_anomaly(
        n_visible: int, n_volumetric_range: Tuple[int, int]
    ) -> bool:
        """
        Check if visual count contradicts volumetric estimate.
        """
        if n_volumetric_range == (0, 0):
            return False  # No volumetric data available

        min_v, max_v = n_volumetric_range

        # If visual count is OUTSIDE the volumetric range, it's an anomaly.
        if n_visible < min_v or n_visible > max_v:
            return True

        return False

    def evaluate(
        self,
        n_visible: int,
        n_volumetric_range: Tuple[int, int],
        residue_ratio: float,
        unexplained_blob_area: float,
        detection_confidence: float,
        current_loop_count: int,
        has_spatial_anomaly: bool = False,
        has_volumetric_anomaly: bool = False,
    ) -> GateDecision:
        """
        Evaluate perception state and decide action.

        Args:
            n_visible: Visual count from CountGD.
            n_volumetric_range: (min, max) count from volumetric estimation.
            residue_ratio: Ratio of unexplained spike energy.
            unexplained_blob_area: Ratio of unexplained blob area.
            detection_confidence: Overall detection confidence.
            current_loop_count: Current number of recursive loops.
            has_spatial_anomaly: Whether spatial anomaly was detected.
            has_volumetric_anomaly: Whether volumetric anomaly was detected.

        Returns:
            GateDecision with action, anomaly type, and reasoning.
        """
        # Safety: Force exit if max loops reached
        if current_loop_count >= self.max_loop_count:
            return GateDecision(
                action="exit",
                anomaly_type=AnomalyType.NONE,
                confidence=detection_confidence,
                rule_applied="MaxLoopSafety",
                reasoning=f"Forced exit: max loops ({self.max_loop_count}) reached.",
                details={
                    "loop_count": current_loop_count,
                    "n_visible": n_visible,
                },
            )

        # Build debug details
        details = {
            "n_visible": n_visible,
            "n_volumetric_range": n_volumetric_range,
            "residue_ratio": residue_ratio,
            "unexplained_blob_area": unexplained_blob_area,
            "detection_confidence": detection_confidence,
            "loop_count": current_loop_count,
        }

        # --- Rule 1: High Confidence PASS ---
        if (
            detection_confidence > self.high_conf
            and unexplained_blob_area < self.unexplained_threshold
            and not has_spatial_anomaly
            and not has_volumetric_anomaly
        ):
            return GateDecision(
                action="exit",
                anomaly_type=AnomalyType.NONE,
                confidence=detection_confidence,
                rule_applied="Rule1_HighConfidencePass",
                reasoning=f"High confidence ({detection_confidence:.2f}) with minimal unexplained area. Accepting count: {n_visible}.",
                details=details,
            )

        # --- Rule 2: Low Confidence FAIL (Noise) ---
        if detection_confidence < self.low_conf and not has_spatial_anomaly:
            return GateDecision(
                action="exit",
                anomaly_type=AnomalyType.NONE,
                confidence=detection_confidence,
                rule_applied="Rule2_LowConfidenceFail",
                reasoning=f"Low confidence ({detection_confidence:.2f}). Treating as noise, using current count: {n_visible}.",
                details=details,
            )

        # --- Anomaly Detection (Hierarchical) ---

        # Check 1: Spatial Anomaly (Residual Spike)
        if has_spatial_anomaly or residue_ratio > self.residue_threshold:
            return GateDecision(
                action="loop",
                anomaly_type=AnomalyType.SPATIAL,
                confidence=detection_confidence,
                rule_applied="Rule3_SpatialAnomaly",
                reasoning=f"Spatial anomaly detected: {residue_ratio:.2%} unexplained spike energy. Triggering SLM for discovery.",
                details=details,
            )

        # Check 2: Volumetric Anomaly (Count out of range)
        if has_volumetric_anomaly:
            min_v, max_v = n_volumetric_range
            return GateDecision(
                action="loop",
                anomaly_type=AnomalyType.VOLUMETRIC,
                confidence=detection_confidence,
                rule_applied="Rule3_VolumetricAnomaly",
                reasoning=f"Volumetric anomaly: N_visible={n_visible} outside range [{min_v}, {max_v}]. Triggering SLM for refinement.",
                details=details,
            )

        # Check 3: High Unexplained Area
        if unexplained_blob_area > self.unexplained_threshold:
            return GateDecision(
                action="loop",
                anomaly_type=AnomalyType.SPATIAL,
                confidence=detection_confidence,
                rule_applied="Rule3_UnexplainedArea",
                reasoning=f"High unexplained blob area: {unexplained_blob_area:.2%}. Triggering SLM.",
                details=details,
            )

        # --- Rule 3: Ambiguous (Default to Loop for first iteration) ---
        if current_loop_count == 0:
            return GateDecision(
                action="loop",
                anomaly_type=AnomalyType.NONE,
                confidence=detection_confidence,
                rule_applied="Rule3_FirstPassAmbiguous",
                reasoning=f"Moderate confidence ({detection_confidence:.2f}). First pass, verifying with SLM.",
                details=details,
            )

        # After first loop, accept if no clear anomaly
        return GateDecision(
            action="exit",
            anomaly_type=AnomalyType.NONE,
            confidence=detection_confidence,
            rule_applied="Rule1_PostLoopAccept",
            reasoning=f"Post-loop: No clear anomaly. Accepting count: {n_visible}.",
            details=details,
        )


if __name__ == "__main__":
    # Quick test
    gate = LogicGate()

    # Test Case 1: High confidence, no anomaly → EXIT
    decision = gate.evaluate(
        n_visible=5,
        n_volumetric_range=(4, 6),
        residue_ratio=0.05,
        unexplained_blob_area=0.03,
        detection_confidence=0.92,
        current_loop_count=0,
    )
    print(f"Test 1: {decision.action} - {decision.rule_applied}")
    print(f"  Reasoning: {decision.reasoning}\n")

    # Test Case 2: Low confidence → EXIT (noise)
    decision = gate.evaluate(
        n_visible=1,
        n_volumetric_range=(0, 0),
        residue_ratio=0.02,
        unexplained_blob_area=0.01,
        detection_confidence=0.25,
        current_loop_count=0,
    )
    print(f"Test 2: {decision.action} - {decision.rule_applied}")
    print(f"  Reasoning: {decision.reasoning}\n")

    # Test Case 3: Spatial anomaly → LOOP
    decision = gate.evaluate(
        n_visible=5,
        n_volumetric_range=(4, 6),
        residue_ratio=0.25,
        unexplained_blob_area=0.03,
        detection_confidence=0.75,
        current_loop_count=0,
        has_spatial_anomaly=True,
    )
    print(f"Test 3: {decision.action} - {decision.rule_applied}")
    print(f"  Reasoning: {decision.reasoning}\n")

    # Test Case 4: Volumetric anomaly → LOOP
    decision = gate.evaluate(
        n_visible=10,
        n_volumetric_range=(4, 6),
        residue_ratio=0.05,
        unexplained_blob_area=0.03,
        detection_confidence=0.80,
        current_loop_count=0,
        has_volumetric_anomaly=True,
    )
    print(f"Test 4: {decision.action} - {decision.rule_applied}")
    print(f"  Reasoning: {decision.reasoning}\n")
