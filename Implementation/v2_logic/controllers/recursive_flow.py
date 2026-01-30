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
_slm_engine: Optional["SLMEngine"] = None
_reid_engine: Optional["ReIDEngine"] = None
_v2e_engine: Optional["V2EEngine"] = None
_vjepa_engine: Optional["VJEPAEngine"] = None


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


def get_reid_engine():
    """Lazy-load ReIDEngine (Phase 3)."""
    global _reid_engine
    if _reid_engine is None:
        try:
            from ..models.reid_engine import ReIDEngine

            _reid_engine = ReIDEngine()
            logger.info("[Engines] ReIDEngine initialized")
        except Exception as e:
            logger.warning("[Engines] Failed to load ReIDEngine: %s", e)
    return _reid_engine


def get_v2e_engine():
    """Lazy-load V2EEngine."""
    global _v2e_engine
    if _v2e_engine is None:
        try:
            from ..models.v2e_engine import V2EEngine

            _v2e_engine = V2EEngine()
            logger.info("[Engines] V2EEngine initialized")
        except Exception as e:
            logger.warning("[Engines] Failed to load V2EEngine: %s", e)
    return _v2e_engine


def get_vjepa_engine():
    """Lazy-load VJEPAEngine."""
    global _vjepa_engine
    if _vjepa_engine is None:
        try:
            from ..models.v_jepa_engine import VJEPAEngine

            _vjepa_engine = VJEPAEngine()
            logger.info("[Engines] VJEPAEngine initialized")
        except Exception as e:
            logger.warning("[Engines] Failed to load VJEPAEngine: %s", e)
    return _vjepa_engine


# =============================================================================
# NODE DEFINITIONS
# =============================================================================


def v2e_sensor_node(state: RecursiveFlowState) -> Dict[str, Any]:
    """
    Generate synthetic events from RGB frame using V2E.
    Produces high-temporal 'spikes' for fusion.
    """
    logger.info("[v2e_sensor_node] Generating events")
    v2e = get_v2e_engine()
    perception = state["perception"]

    if v2e and perception.image is not None:
        timestamp = perception.current_frame_idx / 30.0  # Assume 30 FPS
        events = v2e.generate_events(perception.image, timestamp)

        # Process events into a summary spike map
        h, w = perception.image.shape[:2]
        spike_map = np.zeros((h, w), dtype=np.float32)
        if events is not None and len(events) > 0:
            # Simple summation for the node update
            xs, ys, ps = (
                events[:, 1].astype(int),
                events[:, 2].astype(int),
                events[:, 3],
            )
            mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            np.add.at(spike_map, (ys[mask], xs[mask]), np.abs(ps[mask]))

        updated_perception = perception.model_copy(
            update={
                "v2e_spike_map": spike_map,
                "spike_energy": float(np.sum(spike_map)),
            }
        )
        return {"perception": updated_perception}

    return {}


def vjepa_brain_node(state: RecursiveFlowState) -> Dict[str, Any]:
    """
    Encode frame into V-JEPA latent space.
    Maintains world context for permanence.
    """
    logger.info("[vjepa_brain_node] Encoding latent context")
    vjepa = get_vjepa_engine()
    perception = state["perception"]

    if vjepa and perception.image is not None:
        # Preprocess image to tensor [1, 3, 224, 224] for V-JEPA
        import torch

        img = perception.image
        if img.shape[0] != 224 or img.shape[1] != 224:
            import cv2

            img = cv2.resize(img, (224, 224))

        tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        # V-JEPA expects frames=16 for full encode,
        # but the wrapper 'VJEPAEngine.encode' handles single or multi frames
        # through its vision transformer implementation.
        _ = vjepa.encode(tensor)

        # For now, we don't store the latent in state to save memory,
        # but the engine instance maintains the context.
        return {}

    return {}


# ... (Node Definitions until Fusion)


def vljepa_director_node(state: RecursiveFlowState) -> Dict[str, Any]:
    """
    Generate or update intent list based on V-JEPA latent.
    Acts as the "Sutradara" (Director) of the system.
    """
    ctx = state["ctx"]
    decision = state["decision"]

    current_intent = ctx.main_intent

    # Phase 3: Adaptive Intent Update
    if decision.slm_hypothesis:
        logger.info("[director] SLM Hypothesis received: %s", decision.slm_hypothesis)
        # In a real implementation, we would parse the hypothesis to refine the intent.
        # E.g., "Check behind the pile" -> Add "occluded_pile" to intent?
        # For now, we just log it and potentially flag a specialized search strategy.
        pass

    return (
        {}
    )  # Intent is in Context, which is immutable-ish for main_intent list content but we assume static for now.


def countgd_executor_node(state: RecursiveFlowState) -> Dict[str, Any]:
    """
    Execute zero-shot counting using CountGD.
    Returns N_visible (visual count).
    """
    intent = state["ctx"].main_intent
    logger.info("[countgd_executor_node] Counting objects matching intent: %s", intent)

    engine = get_countgd_engine()
    perception = state["perception"]

    if engine and perception.image is not None:
        # Real CountGD call
        count_val, detections = engine.count(perception.image, intent)

        updated_perception = perception.model_copy(
            update={
                "n_visible": count_val,
                "raw_detections": detections,
            }
        )
        return {"perception": updated_perception}

    return {}


def sam2_depth_node(state: RecursiveFlowState) -> Dict[str, Any]:
    """
    Segment objects (SAM2) and estimate depth (DepthAnything V2).
    Returns 3D point cloud summary and volumetric count range.
    """
    logger.info(
        "[sam2_depth_node] Generating point cloud for frame %d",
        state["perception"].current_frame_idx,
    )

    seg_engine = get_segmentation_engine()
    depth_engine = get_depth_engine()
    perception = state["perception"]
    ctx = state["ctx"]

    if perception.image is None:
        return {}

    updates = {}
    image = perception.image

    # 1. Depth Estimation
    depth_map = None
    if depth_engine:
        depth_res = depth_engine.estimate_depth(image)
        if depth_res.has_depth:
            depth_map = depth_res.depth_map
            updates["depth_map_stats"] = depth_res.stats

    # 2. Segmentation
    masks = []
    if seg_engine:
        seg_res = seg_engine.segment_frame(image)
        masks = seg_res.masks
        updates["masks"] = masks

    # 3. Volumetric Estimation
    if depth_map is not None and len(masks) > 0:
        total_volume = 0.0
        from ..utils.math_utils import MathUtils

        for mask in masks:
            vol = MathUtils.estimate_volume_heuristic(
                depth_map=depth_map,
                mask=mask.astype(bool),
                fx=500.0,
                fy=500.0,  # Generic focal lengths
            )
            total_volume += vol * ctx.depth_scale_factor  # Scale to meters

        min_c, max_c = MathUtils.lattice_counting(total_volume, ctx.unit_volume_prior)
        updates["n_volumetric_range"] = (min_c, max_c)
        updates["point_cloud_summary"] = {"total_volume": float(total_volume)}

    if "n_volumetric_range" not in updates:
        updates["n_volumetric_range"] = (0, 0)

    updated_perception = perception.model_copy(update=updates)
    return {"perception": updated_perception}


def fusion_engine_node(state: RecursiveFlowState) -> Dict[str, Any]:
    """
    Fuse spike data with SAM2 masks and TRACK objects.
    Implements Re-ID to maintain consistent counts across loops.
    """
    logger.info("[fusion_engine_node] Fusing sensor data & Tracking")

    fusion_engine = get_fusion_engine()
    reid_engine = get_reid_engine()
    perception = state["perception"]

    updates = {}

    # 1. Fusion (Spike-Mask)
    if fusion_engine and perception.v2e_spike_map is not None:
        masks = perception.masks if perception.masks else []
        result = fusion_engine.fuse_spike_mask(
            spike_map=perception.v2e_spike_map,
            masks=masks,
            n_visible=perception.n_visible,
            n_volumetric_range=perception.n_volumetric_range,
        )
        updates["residual_spike_energy"] = result.residual_spike_energy
        updates["unexplained_blobs"] = result.unexplained_blobs

    # 2. Re-Identification (Phase 3)
    if reid_engine:
        current_detections = perception.raw_detections
        matched, new_dets = reid_engine.match_detections(
            current_detections, perception.current_frame_idx
        )

        # Combine for final tracked list
        all_tracked = matched + new_dets
        updates["tracked_objects"] = all_tracked

        # Update N_visible based on unique tracks if needed,
        # but usually CountGD gives the raw count.
        # If we loop, we rely on the tracked list size?
        # For now, let's keep n_visible as CountGD's output,
        # but OutputAccumulator should use tracked objects count.

    updated_perception = perception.model_copy(update=updates)
    return {"perception": updated_perception}


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

        # Determine anomalies dynamically (Phase 4)
        is_vol_anomaly = False
        if hasattr(engine, "check_volumetric_anomaly"):
            is_vol_anomaly = engine.check_volumetric_anomaly(
                perception.n_visible, perception.n_volumetric_range
            )

        # Determine anomalies (simplified check)
        gate_decision = engine.evaluate(
            n_visible=perception.n_visible,
            n_volumetric_range=perception.n_volumetric_range,
            residue_ratio=residue_ratio,
            unexplained_blob_area=0.0,  # Placeholder
            detection_confidence=0.9,  # Placeholder
            current_loop_count=decision.loop_count,
            has_spatial_anomaly=residue_ratio > 0.15,  # Example threshold
            has_volumetric_anomaly=is_vol_anomaly,
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


def interpolation_node(state: RecursiveFlowState) -> Dict[str, Any]:
    """
    State Interpolation.
    Projects previous state/hypothesis to current frame coordinates.
    """
    logger.info("[interpolation_node] Interpolating state to current frame")

    decision = state["decision"]

    # Phase 3 Logic:
    # If camera moved (which we don't track explicitly yet),
    # we would transform tracked_objects bboxes here.
    # For now, we assume static camera or essentially 'pass-through'
    # to prepare for the next loop.

    new_decision = decision.model_copy(
        update={
            "loop_count": decision.loop_count + 1,
            "status": "looping",
        }
    )
    return {"decision": new_decision}


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
