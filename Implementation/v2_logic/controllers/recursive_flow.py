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
from typing import Any, Dict, Literal

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
# NODE DEFINITIONS (Skeleton - Placeholder Logic)
# =============================================================================


def v2e_sensor_node(state: RecursiveFlowState) -> Dict[str, Any]:
    """
    Convert RGB frame to event spikes using V2E.
    Runs in parallel with vjepa_brain_node.

    Phase 0: Placeholder - Returns zero spike energy.
    """
    logger.info(
        "[v2e_sensor_node] Processing frame %d", state["perception"].current_frame_idx
    )

    # Placeholder: No actual spike generation
    updated_perception = state["perception"].model_copy(
        update={
            "spike_energy": 0.0,
            "residual_spike_energy": 0.0,
        }
    )
    return {"perception": updated_perception}


def vjepa_brain_node(state: RecursiveFlowState) -> Dict[str, Any]:
    """
    Process frame through V-JEPA for temporal memory.
    Generates latent representation for the Director.

    Phase 0: Placeholder - No actual V-JEPA inference.
    """
    logger.info(
        "[vjepa_brain_node] Generating latent for frame %d",
        state["perception"].current_frame_idx,
    )

    # Placeholder: V-JEPA latent would be stored here
    # For now, we just pass through
    return {}


def vljepa_director_node(state: RecursiveFlowState) -> Dict[str, Any]:
    """
    Generate or update intent list based on V-JEPA latent.
    Acts as the "Sutradara" (Director) of the system.

    Phase 0: Placeholder - Uses main_intent from GlobalContext.
    """
    logger.info("[vljepa_director_node] Current intent: %s", state["ctx"].main_intent)

    # In Phase 0, we just echo the main intent
    # Later, this will update based on SLM feedback
    return {}


def countgd_executor_node(state: RecursiveFlowState) -> Dict[str, Any]:
    """
    Execute zero-shot counting using CountGD.
    Returns N_visible (visual count).

    Phase 0: Placeholder - Returns mock count.
    """
    intent = state["ctx"].main_intent
    logger.info("[countgd_executor_node] Counting objects matching intent: %s", intent)

    # Placeholder: Mock detection
    updated_perception = state["perception"].model_copy(
        update={
            "n_visible": 0,  # Placeholder
            "raw_detections": [],
        }
    )
    return {"perception": updated_perception}


def sam2_depth_node(state: RecursiveFlowState) -> Dict[str, Any]:
    """
    Segment objects (SAM2) and estimate depth (DepthAnything V2).
    Returns 3D point cloud summary and volumetric count range.

    Phase 0: Placeholder - Returns empty point cloud.
    """
    logger.info(
        "[sam2_depth_node] Generating point cloud for frame %d",
        state["perception"].current_frame_idx,
    )

    # Placeholder: Mock point cloud
    updated_perception = state["perception"].model_copy(
        update={
            "depth_map_stats": {"has_depth": False},
            "point_cloud_summary": {"num_points": 0},
            "n_volumetric_range": (0, 0),
        }
    )
    return {"perception": updated_perception}


def fusion_engine_node(state: RecursiveFlowState) -> Dict[str, Any]:
    """
    Fuse spike data with SAM2 masks to detect residual spikes.
    Implements Motion Compensation to filter camera jitter.

    Phase 0: Placeholder - No actual fusion.
    """
    logger.info("[fusion_engine_node] Fusing sensor data")

    # Calculate residual = total spike - masked spike (placeholder)
    updated_perception = state["perception"].model_copy(
        update={
            "residual_spike_energy": 0.0,
            "unexplained_blobs": [],
        }
    )
    return {"perception": updated_perception}


def logic_gate_node(state: RecursiveFlowState) -> Dict[str, Any]:
    """
    Primary decision gate. Checks for anomalies and decides:
    - EXIT: Confident count, no anomalies.
    - LOOP: Ambiguity detected, trigger SLM.

    Implements the "Math Guards":
    1. Spatial Anomaly: High residual spike.
    2. Volumetric Anomaly: N_visible outside N_volumetric range.
    3. Physical Constraint: Volume exceeds physical bounds.

    Phase 0: Placeholder - Always returns EXIT.
    """
    perception = state["perception"]
    decision = state["decision"]
    ctx = state["ctx"]

    logger.info(
        "[logic_gate_node] N_visible=%d, Residual=%.2f, Loop=%d/%d",
        perception.n_visible,
        perception.residual_spike_energy,
        decision.loop_count,
        ctx.max_loop_count,
    )

    # Placeholder logic: Always exit in Phase 0
    # Real logic will be implemented in Phase 2
    new_decision = decision.model_copy(
        update={
            "status": "exit",
            "anomaly_type": "none",
            "logic_gate_result": {
                "rule_applied": "Phase0_AlwaysExit",
                "confidence": 1.0,
                "action": "exit",
            },
        }
    )

    # Update output accumulator
    new_output = state["output"].model_copy(
        update={
            "final_count": perception.n_visible,
            "confidence": 1.0,
            "processing_log": state["output"].processing_log
            + ["[logic_gate] Phase 0: Direct exit"],
        }
    )

    return {"decision": new_decision, "output": new_output}


def targeted_slm_node(state: RecursiveFlowState) -> Dict[str, Any]:
    """
    Targeted SLM for ambiguity resolution.
    Only triggered when Logic Gate detects anomalies.

    Phase 0: Placeholder - Not triggered.
    """
    logger.info("[targeted_slm_node] SLM reasoning triggered")

    decision = state["decision"]
    new_decision = decision.model_copy(
        update={
            "slm_triggered": True,
            "slm_reasoning": "Phase 0: Placeholder reasoning",
            "slm_hypothesis": None,
        }
    )
    return {"decision": new_decision}


def interpolation_node(state: RecursiveFlowState) -> Dict[str, Any]:
    """
    State Interpolation using V-JEPA predictions.
    Projects SLM decisions onto current frame coordinates.

    Phase 0: Placeholder - No interpolation.
    """
    logger.info("[interpolation_node] Interpolating state to current frame")

    decision = state["decision"]
    new_decision = decision.model_copy(
        update={
            "loop_count": decision.loop_count + 1,
            "status": "looping",
        }
    )
    return {"decision": new_decision}


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================


def route_after_logic_gate(
    state: RecursiveFlowState,
) -> Literal["exit", "targeted_slm_node"]:
    """
    Conditional edge: Route based on Logic Gate decision.
    """
    decision = state["decision"]

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
