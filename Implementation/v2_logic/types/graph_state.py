"""
Graph State Models

Pydantic V2 models for the Recursive Intent LangGraph state architecture.
Implements the Scoped State Design to prevent "God Object" anti-pattern.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : GraphStateModels (this module)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <ContextModels>   → GlobalContext                                        │
  │  <StateModels>     → PerceptionState | DecisionState | OutputAccumulator  │
  │  <RootContainer>   → RecursiveFlowState                                   │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <BaseModel>   ← from pydantic (Validation & Serialization)               │
  │  <TypedDict>   ← from typing (LangGraph State Container)                  │
  │  <Literal>     ← from typing (Status Enum)                                │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : str, int, float, List, Dict, Tuple, Optional, Literal

Production Rules:
  GraphStateModels     → imports + <ContextModels> + <StateModels> + <RootContainer>
  <ContextModels>      → GlobalContext
  <StateModels>        → PerceptionState | DecisionState | OutputAccumulator
  <RootContainer>      → RecursiveFlowState(TypedDict)
═══════════════════════════════════════════════════════════════════════════════

Pattern: Data Transfer Object (DTO)
- Encapsulates state data for transfer between LangGraph nodes.
- Provides validation and serialization via Pydantic.
"""

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, TypedDict

from pydantic import BaseModel, Field


def merge_perception(
    old: "PerceptionState", new: "PerceptionState"
) -> "PerceptionState":
    """Reducer to merge perception state updates."""
    # We use model_copy with update to merge the pydantic model fields
    # Fields in 'new' that are not None/default will override 'old'
    updates = new.model_dump(exclude_unset=True)
    return old.model_copy(update=updates)


# =============================================================================
# CONTEXT MODELS (Session-Level)
# =============================================================================


class GlobalContext(BaseModel):
    """
    Session-level context that persists across the entire workflow.
    Immutable after initialization.

    Pattern: Value Object
    - Represents session identity and initial configuration.
    """

    session_id: str = Field(..., description="Unique identifier for this session")
    main_intent: List[str] = Field(
        default_factory=list, description="List of target object labels to count"
    )
    start_time: float = Field(..., description="Timestamp of session start (epoch)")
    max_loop_count: int = Field(
        default=3, description="Maximum recursive loops before forced exit"
    )
    # Phase 4: Volumetric Priors
    unit_volume_prior: float = Field(
        default=0.001, description="Estimated volume of a single target object (m^3)"
    )
    depth_scale_factor: float = Field(
        default=10.0, description="Conversion factor from relative depth to meters"
    )


# =============================================================================
# PERCEPTION STATE (Sensor-Level)
# =============================================================================


class PerceptionState(BaseModel):
    """
    State owned by perception components: V2E, SAM2, DepthAnything, CountGD.
    Updated every frame by sensor nodes.

    Pattern: Data Transfer Object (DTO)
    - Carries raw sensor data between nodes.
    """

    current_frame_idx: int = Field(default=0, description="Current frame index")
    image: Optional[Any] = Field(
        default=None, description="Current RGB frame (numpy array)"
    )
    last_frame: Optional[Any] = Field(
        default=None, description="Previous RGB frame (numpy array)"
    )
    masks: Optional[Any] = Field(
        default=None, description="SAM2 masks (list of arrays)"
    )
    v2e_spike_map: Optional[Any] = Field(
        default=None, description="Event spike map (numpy array)"
    )

    # --- CountGD Output ---
    n_visible: int = Field(
        default=0, description="Number of visually detected objects (N_visible)"
    )
    raw_detections: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of detection dicts with bbox, confidence, label",
    )

    # --- SAM2 + Depth Output ---
    depth_map_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Depth map statistics: mean_depth, min, max, has_depth",
    )
    point_cloud_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Point cloud summary: num_points, volume_estimate, bounds",
    )
    n_volumetric_range: Tuple[int, int] = Field(
        default=(0, 0),
        description="Estimated count range [min, max] based on volume",
    )

    # --- V2E Output ---
    spike_energy: float = Field(
        default=0.0, description="Total spike energy in current frame"
    )
    residual_spike_energy: float = Field(
        default=0.0,
        description="Spike energy outside detected mask areas (Anomaly Signal)",
    )

    # --- Fusion Output ---
    unexplained_blobs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of unexplained blob regions with coordinates",
    )
    tracked_objects: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of tracked objects with consistent IDs across loops",
    )


# =============================================================================
# DECISION STATE (Logic-Level)
# =============================================================================


class DecisionState(BaseModel):
    """
    State owned by Logic Gate and Targeted SLM.
    Tracks the decision-making process and loop status.

    Pattern: State Object
    - Represents the current state of the decision machine.
    """

    status: Literal["processing", "looping", "exit"] = Field(
        default="processing", description="Current workflow status"
    )

    # --- Anomaly Detection ---
    anomaly_type: Optional[Literal["spatial", "volumetric", "physics", "none"]] = Field(
        default=None, description="Type of anomaly detected by Logic Gate"
    )
    anomaly_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Details about the anomaly: location, magnitude, etc.",
    )

    # --- Logic Gate Result ---
    logic_gate_result: Dict[str, Any] = Field(
        default_factory=dict,
        description="Gate decision: rule_applied, confidence, action",
    )

    # --- SLM Reasoning (Slow Path) ---
    slm_triggered: bool = Field(
        default=False, description="Whether Targeted SLM was invoked"
    )
    slm_reasoning: Optional[str] = Field(
        default=None, description="SLM's reasoning output"
    )
    slm_hypothesis: Optional[str] = Field(
        default=None, description="SLM's hypothesis for intent update"
    )

    # --- Loop Control ---
    loop_count: int = Field(default=0, description="Number of recursive loops so far")
    loop_trigger_reason: Optional[str] = Field(
        default=None, description="Reason for triggering the loop"
    )


# =============================================================================
# OUTPUT ACCUMULATOR (Result-Level)
# =============================================================================


class OutputAccumulator(BaseModel):
    """
    Accumulates final results across frames.

    Pattern: Accumulator
    - Collects and aggregates results for final output.
    """

    final_count: int = Field(default=0, description="Final object count")
    confidence: float = Field(
        default=0.0, description="Confidence score of final count"
    )
    detections: List[Dict[str, Any]] = Field(
        default_factory=list, description="Final list of detected objects with details"
    )
    processing_log: List[str] = Field(
        default_factory=list, description="Log of processing steps for debugging"
    )


# =============================================================================
# ROOT STATE CONTAINER (LangGraph)
# =============================================================================


class RecursiveFlowState(TypedDict):
    """
    The root container for LangGraph StateGraph.
    Uses TypedDict for LangGraph compatibility.

    Pattern: Composite
    - Aggregates all sub-states into a single container.

    Note: LangGraph requires TypedDict, not Pydantic models, at the root level.
    """

    ctx: GlobalContext
    perception: Annotated[PerceptionState, merge_perception]
    decision: DecisionState
    output: OutputAccumulator


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_initial_state(
    session_id: str,
    main_intent: List[str],
    start_time: float,
) -> RecursiveFlowState:
    """
    Factory function to create a fresh initial state for each session.

    Args:
        session_id: Unique identifier for this session.
        main_intent: List of target object labels to count.
        start_time: Timestamp of session start (epoch).

    Returns:
        RecursiveFlowState: Initialized state container.
    """
    return RecursiveFlowState(
        ctx=GlobalContext(
            session_id=session_id,
            main_intent=main_intent,
            start_time=start_time,
        ),
        perception=PerceptionState(),
        decision=DecisionState(),
        output=OutputAccumulator(),
    )
