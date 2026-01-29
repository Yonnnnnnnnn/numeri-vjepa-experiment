"""
Recursive Flow Controller

LangGraph StateGraph implementation for the Recursive Intent system.
Orchestrates the flow between perception, fusion, decision, and recursion nodes.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : RecursiveFlowController (this module)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <SensorNodes>     → v2e_sensor_node | vjepa_brain_node                   │
  │  <DirectorNodes>   → vljepa_director_node                                 │
  │  <ExecutorNodes>   → countgd_executor_node | sam2_depth_node              │
  │  <FusionNodes>     → fusion_engine_node                                   │
  │  <DecisionNodes>   → logic_gate_node | targeted_slm_node                  │
  │  <UtilityNodes>    → interpolation_node                                   │
  │  <GraphBuilder>    → build_recursive_graph                                │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <StateGraph>              ← from langgraph.graph (Graph Builder)         │
  │  <RecursiveFlowState>      ← from types.graph_state (State Container)     │
  │  <END>                     ← from langgraph.graph (Terminal Node)         │
  │  --- Phase 1 Engines ---                                                  │
  │  <SegmentationEngine>      ← from models.segmentation_engine              │
  │  <DepthEngine>             ← from models.depth_engine                     │
  │  <CountGDEngine>           ← from models.count_gd_engine                  │
  │  <FusionEngineV2>          ← from models.fusion_engine_v2                 │
  │  <LogicGate>               ← from models.logic_gate                       │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : "v2e_sensor", "vjepa_brain", "vljepa_director", etc.

Production Rules:
  RecursiveFlowController → imports + <NodeDefinitions> + <GraphBuilder>
  <NodeDefinitions>       → <SensorNodes> + <DirectorNodes> + <ExecutorNodes> +
                            <FusionNodes> + <DecisionNodes> + <UtilityNodes>
  <GraphBuilder>          → StateGraph + add_node* + add_edge* + compile
═══════════════════════════════════════════════════════════════════════════════

Pattern: Builder
- Constructs the LangGraph StateGraph step by step.
- Allows flexible node and edge configuration.
"""

import logging
from typing import Any, Dict, Literal, Optional

import numpy as np

from langgraph.graph import END, START, StateGraph

from ..types.graph_state import (
    DecisionState,
    GlobalContext,
    OutputAccumulator,
    PerceptionState,
    RecursiveFlowState,
)

logger = logging.getLogger(__name__)

# =============================================================================
# LAZY-LOADED ENGINE INSTANCES (Phase 1)
# =============================================================================

# ... (imports remain)

# =============================================================================
# LAZY-LOADED ENGINE INSTANCES (Phase 1 & 2)
# =============================================================================

# Global engine instances (initialized on first use)
_segmentation_engine: Optional["SegmentationEngine"] = None
_depth_engine: Optional["DepthEngine"] = None
_countgd_engine: Optional["CountGDEngine"] = None
_fusion_engine: Optional["FusionEngineV2"] = None
_logic_gate: Optional["LogicGate"] = None
_slm_engine: Optional["SLMEngine"] = None  # New in Phase 2


def get_segmentation_engine():
    """Lazy-load SegmentationEngine (SAM2)."""
    global _segmentation_engine
    if _segmentation_engine is None:
        try:
            from ..models.segmentation_engine import SegmentationEngine

            _segmentation_engine = SegmentationEngine()
            logger.info("[Engines] SegmentationEngine initialized")
        except Exception as e:
            logger.warning("[Engines] Failed to load SegmentationEngine: %s", e)
    return _segmentation_engine


def get_depth_engine():
    """Lazy-load DepthEngine (DepthAnything V2)."""
    global _depth_engine
    if _depth_engine is None:
        try:
            from ..models.depth_engine import DepthEngine

            _depth_engine = DepthEngine(encoder="vits")
            logger.info("[Engines] DepthEngine initialized")
        except Exception as e:
            logger.warning("[Engines] Failed to load DepthEngine: %s", e)
    return _depth_engine


def get_countgd_engine():
    """Lazy-load CountGDEngine."""
    global _countgd_engine
    if _countgd_engine is None:
        try:
            from ..models.count_gd_engine import CountGDEngine

            _countgd_engine = CountGDEngine()
            logger.info("[Engines] CountGDEngine initialized")
        except Exception as e:
            logger.warning("[Engines] Failed to load CountGDEngine: %s", e)
    return _countgd_engine


def get_fusion_engine():
    """Lazy-load FusionEngineV2."""
    global _fusion_engine
    if _fusion_engine is None:
        try:
            from ..models.fusion_engine_v2 import FusionEngineV2

            _fusion_engine = FusionEngineV2()
            logger.info("[Engines] FusionEngineV2 initialized")
        except Exception as e:
            logger.warning("[Engines] Failed to load FusionEngineV2: %s", e)
    return _fusion_engine


def get_logic_gate():
    """Lazy-load LogicGate."""
    global _logic_gate
    if _logic_gate is None:
        try:
            from ..models.logic_gate import LogicGate

            _logic_gate = LogicGate()
            logger.info("[Engines] LogicGate initialized")
        except Exception as e:
            logger.warning("[Engines] Failed to load LogicGate: %s", e)
    return _logic_gate


def get_slm_engine():
    """Lazy-load SLMEngine (Phase 2)."""
    global _slm_engine
    if _slm_engine is None:
        try:
            from ..models.slm_engine import SLMEngine

            _slm_engine = SLMEngine()
            logger.info("[Engines] SLMEngine initialized")
        except Exception as e:
            logger.warning("[Engines] Failed to load SLMEngine: %s", e)
    return _slm_engine


# ... (Node Definitions until Fusion)


def fusion_engine_node(state: RecursiveFlowState) -> Dict[str, Any]:
    """
    Fuse spike data with SAM2 masks to detect residual spikes.
    Implements Motion Compensation to filter camera jitter.
    """
    logger.info("[fusion_engine_node] Fusing sensor data")

    engine = get_fusion_engine()
    perception = state["perception"]

    if engine and perception.v2e_spike_map is not None:
        # Assuming masks are available (mock or real)
        # In Phase 2, we would use real masks if SAM2 ran
        # For now, we use a placeholder or previous masks

        # Convert list of masks to list of numpy arrays if needed
        masks = perception.masks if perception.masks else []

        result = engine.fuse_spike_mask(
            spike_map=perception.v2e_spike_map,
            masks=masks,
            n_visible=perception.n_visible,
            n_volumetric_range=perception.n_volumetric_range,
        )

        updated_perception = perception.model_copy(
            update={
                "residual_spike_energy": result.residual_spike_energy,
                "unexplained_blobs": result.unexplained_blobs,
            }
        )

        # Also update decision state context with fusion results
        return {"perception": updated_perception}

    return {}


def logic_gate_node(state: RecursiveFlowState) -> Dict[str, Any]:
    """
    Primary decision gate. Checks for anomalies.
    """
    perception = state["perception"]
    decision = state["decision"]
    ctx = state["ctx"]

    engine = get_logic_gate()
    if engine:
        # Calculate derived metrics for the gate
        # In a real run, these would come from Fusion Result, but we compute them here for the call
        total_energy = perception.spike_energy if perception.spike_energy > 0 else 1.0
        residue_ratio = perception.residual_spike_energy / total_energy

        # Determine anomalies (simplified check)
        gate_decision = engine.evaluate(
            n_visible=perception.n_visible,
            n_volumetric_range=perception.n_volumetric_range,
            residue_ratio=residue_ratio,
            unexplained_blob_area=0.0,  # Placeholder
            detection_confidence=0.9,  # Placeholder
            current_loop_count=decision.loop_count,
            has_spatial_anomaly=residue_ratio > 0.15,  # Example threshold
            has_volumetric_anomaly=False,
        )

        new_decision = decision.model_copy(
            update={
                "status": gate_decision.action,
                "anomaly_type": gate_decision.anomaly_type.value,
                "logic_gate_result": {
                    "rule_applied": gate_decision.rule_applied,
                    "confidence": gate_decision.confidence,
                    "action": gate_decision.action,
                    "reasoning": gate_decision.reasoning,
                },
            }
        )

        # Log decision
        logger.info(
            "[logic_gate_node] Decision: %s (Rule: %s)",
            gate_decision.action,
            gate_decision.rule_applied,
        )

        return {"decision": new_decision}

    return {}


def targeted_slm_node(state: RecursiveFlowState) -> Dict[str, Any]:
    """
    Targeted SLM for ambiguity resolution.
    Only triggered when Logic Gate detects anomalies.
    """
    logger.info("[targeted_slm_node] SLM reasoning triggered")

    engine = get_slm_engine()
    decision = state["decision"]
    perception = state["perception"]

    if engine:
        # Prepare context for SLM
        context = {
            "n_visible": perception.n_visible,
            "n_volumetric_range": perception.n_volumetric_range,
            "residue_ratio": 0.2,  # Placeholder or from state
        }

        # Use last frame
        image = (
            perception.last_frame
            if perception.last_frame is not None
            else np.zeros((480, 640, 3), dtype=np.uint8)
        )

        result = engine.generate_reasoning(
            image=image, anomaly_type=decision.anomaly_type, context=context
        )

        new_decision = decision.model_copy(
            update={
                "slm_triggered": True,
                "slm_reasoning": result.reasoning_text,
                "slm_hypothesis": result.hypothesis,
            }
        )
        return {"decision": new_decision}

    return {}


def route_after_logic_gate(
    state: RecursiveFlowState,
) -> Literal["exit", "targeted_slm_node"]:
    """
    Conditional edge: Route based on Logic Gate decision.
    """
    decision = state["decision"]

    logger.info("[route] Checking decision status: %s", decision.status)

    if decision.status == "exit":
        return "exit"
    elif decision.loop_count >= state["ctx"].max_loop_count:
        logger.warning("[route] Max loop count reached, forcing exit")
        return "exit"
    else:
        return "targeted_slm_node"


# =============================================================================
# GRAPH BUILDER
# =============================================================================


def build_recursive_graph() -> StateGraph:
    """
    Build the LangGraph StateGraph for Recursive Intent.

    Returns:
        Compiled StateGraph ready for execution.

    Architecture:
    ```
    START
      ├─(parallel)─> v2e_sensor_node ────────────────────┐
      └─(parallel)─> vjepa_brain_node                    │
                            │                            │
                            v                            │
                     vljepa_director_node                │
                       ├───────────────┬─────────────────┤
                       v               v                 v
              countgd_executor   sam2_depth         (v2e output)
                       │               │                 │
                       └───────────────┴─────────────────┘
                                       │
                                       v
                              fusion_engine_node
                                       │
                                       v
                                logic_gate_node
                                       │
                          ┌────────────┴────────────┐
                          v                         v
                        (exit)              targeted_slm_node
                          │                         │
                          v                         v
                         END               interpolation_node
                                                    │
                                                    v
                                           vljepa_director_node (loop)
    ```
    """
    # Initialize the graph with the state schema
    graph = StateGraph(RecursiveFlowState)

    # --- Add Nodes ---
    graph.add_node("v2e_sensor_node", v2e_sensor_node)
    graph.add_node("vjepa_brain_node", vjepa_brain_node)
    graph.add_node("vljepa_director_node", vljepa_director_node)
    graph.add_node("countgd_executor_node", countgd_executor_node)
    graph.add_node("sam2_depth_node", sam2_depth_node)
    graph.add_node("fusion_engine_node", fusion_engine_node)
    graph.add_node("logic_gate_node", logic_gate_node)
    graph.add_node("targeted_slm_node", targeted_slm_node)
    graph.add_node("interpolation_node", interpolation_node)

    # --- Add Edges ---

    # Parallel sensor paths from START
    graph.add_edge(START, "v2e_sensor_node")
    graph.add_edge(START, "vjepa_brain_node")

    # V-JEPA → Director
    graph.add_edge("vjepa_brain_node", "vljepa_director_node")

    # Director → Parallel Executors
    graph.add_edge("vljepa_director_node", "countgd_executor_node")
    graph.add_edge("vljepa_director_node", "sam2_depth_node")

    # All paths converge at Fusion
    graph.add_edge("v2e_sensor_node", "fusion_engine_node")
    graph.add_edge("countgd_executor_node", "fusion_engine_node")
    graph.add_edge("sam2_depth_node", "fusion_engine_node")

    # Fusion → Logic Gate
    graph.add_edge("fusion_engine_node", "logic_gate_node")

    # Logic Gate → Conditional Edge
    graph.add_conditional_edges(
        "logic_gate_node",
        route_after_logic_gate,
        {
            "exit": END,
            "targeted_slm_node": "targeted_slm_node",
        },
    )

    # SLM → Interpolation → Director (Recursive Loop)
    graph.add_edge("targeted_slm_node", "interpolation_node")
    graph.add_edge("interpolation_node", "vljepa_director_node")

    logger.info("[build_recursive_graph] Graph topology:")
    try:
        graph.get_graph().print_ascii()
    except Exception as e:
        logger.warning("Could not print graph: %s", e)

    return graph.compile()


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================


def create_recursive_flow_app():
    """
    Convenience function to create a compiled LangGraph app.

    Returns:
        Compiled LangGraph app ready for invoke().
    """
    return build_recursive_graph()
