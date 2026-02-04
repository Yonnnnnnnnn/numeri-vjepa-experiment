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
  │  <CountVidEngine>           ← from models.count_vid_engine                │
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

from langgraph.graph import END, START, StateGraph  # pylint: disable=import-error

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
_countvid_engine: Optional["CountVidEngine"] = None
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


def get_countvid_engine():
    """Lazy-load CountVidEngine."""
    global _countvid_engine
    if _countvid_engine is None:
        try:
            from ..models.count_vid_engine import CountVidEngine

            _countvid_engine = CountVidEngine()
            logger.info("[Engines] CountVidEngine initialized")
        except Exception as e:
            logger.warning("[Engines] Failed to load CountVidEngine: %s", e)
    return _countvid_engine


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

            img = cv2.resize(img, (224, 224))  # pylint: disable=no-member

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

    Implements:
    - Discovery Loop: Parse SLM hypothesis for new object labels
    - Refinement Loop: Adjust sensitivity or set PointBeam ROI
    """
    ctx = state["ctx"]
    decision = state["decision"]
    perception = state["perception"]

    # Start with main_intent if active_intent is empty (first run)
    current_intent = (
        perception.active_intent if perception.active_intent else list(ctx.main_intent)
    )
    sensitivity = perception.sensitivity_modifier
    focus_roi = perception.focus_roi

    updates = {}

    # Phase 9: Adaptive Intent Update from SLM Hypothesis
    if decision.slm_hypothesis:
        hypothesis = decision.slm_hypothesis.lower()
        logger.info("[director] SLM Hypothesis received: %s", decision.slm_hypothesis)

        # --- DISCOVERY LOOP: Parse for new object labels ---
        # Common object keywords to look for
        discovery_keywords = [
            "ball",
            "sphere",
            "bottle",
            "mug",
            "glass",
            "plate",
            "bowl",
            "hand",
            "finger",
            "arm",
            "person",
            "box",
            "container",
        ]

        discovered_labels = []
        for keyword in discovery_keywords:
            if keyword in hypothesis and keyword not in current_intent:
                discovered_labels.append(keyword)

        if discovered_labels:
            # Type Guard: Ensure we're appending strings, not lists
            for label in discovered_labels:
                if isinstance(label, str) and label not in current_intent:
                    current_intent.append(label)
            logger.info(
                "[director] DISCOVERY LOOP: Added new labels: %s", discovered_labels
            )
            updates["active_intent"] = current_intent

        # --- REFINEMENT LOOP: Parse for occlusion/sensitivity hints ---
        refinement_hints = [
            "occluded",
            "hidden",
            "behind",
            "blocked",
            "covered",
            "overlap",
        ]
        needs_refinement = any(hint in hypothesis for hint in refinement_hints)

        if needs_refinement:
            # Increase sensitivity for next pass
            new_sensitivity = min(sensitivity * 1.2, 2.0)  # Cap at 2x
            updates["sensitivity_modifier"] = new_sensitivity
            logger.info(
                "[director] REFINEMENT LOOP: Sensitivity adjusted to %.2f",
                new_sensitivity,
            )

            # If "corner" or spatial hint is mentioned, set PointBeam ROI
            if "corner" in hypothesis or "edge" in hypothesis:
                # Example: Focus on bottom-right quadrant
                h, w = (
                    perception.image.shape[:2]
                    if perception.image is not None
                    else (480, 640)
                )
                focus_roi = (w // 2, h // 2, w, h)  # Bottom-right quadrant
                updates["focus_roi"] = focus_roi
                logger.info("[director] POINTBEAM: Set focus ROI to %s", focus_roi)
            elif "left" in hypothesis:
                h, w = (
                    perception.image.shape[:2]
                    if perception.image is not None
                    else (480, 640)
                )
                focus_roi = (0, 0, w // 2, h)  # Left half
                updates["focus_roi"] = focus_roi
                logger.info("[director] POINTBEAM: Set focus ROI to %s", focus_roi)

        # --- VOLUMETRIC AUTO-CALIBRATION ---
        # If we're in a volumetric anomaly loop and SLM confirms the visual count,
        # we should recalibrate the unit volume estimate.
        if decision.anomaly_type == "volumetric":
            # Check if SLM is confirming the visual count
            import re

            # Pattern: "3 objects" or "three objects" or "I see 3"
            count_match = re.search(r"(\d+)\s+object", hypothesis)
            if count_match:
                slm_confirmed_count = int(count_match.group(1))
                if (
                    slm_confirmed_count == perception.n_visible
                    and perception.n_visible > 0
                ):
                    # SLM agrees with visual count -> recalibrate unit volume
                    # Formula: new_unit_volume = total_observed_volume / n_visible
                    total_vol = perception.total_observed_volume
                    if total_vol > 0:
                        new_unit_volume = total_vol / perception.n_visible
                        updates["estimated_unit_volume"] = new_unit_volume
                        logger.info(
                            "[director] VOLUMETRIC CALIBRATION: SLM confirmed count=%d, "
                            "recalibrating unit_volume = %.6f m^3",
                            perception.n_visible,
                            new_unit_volume,
                        )

    # Always ensure active_intent is populated (even if no changes)
    if "active_intent" not in updates and not perception.active_intent:
        updates["active_intent"] = list(ctx.main_intent)

    if updates:
        updated_perception = perception.model_copy(update=updates)
        return {"perception": updated_perception}

    return {}


def countvid_executor_node(state: RecursiveFlowState) -> Dict[str, Any]:
    """
    Execute zero-shot counting using CountVid.
    Returns N_visible (visual count).

    Supports:
    - Discovery Loop: Uses active_intent (dynamic, updated by director)
    - Refinement Loop: Applies sensitivity_modifier and focus_roi (PointBeam)
    """
    perception = state["perception"]
    ctx = state["ctx"]

    # Use active_intent if available, fallback to main_intent
    intent = perception.active_intent if perception.active_intent else ctx.main_intent
    logger.info("[countvid_executor_node] Counting objects matching intent: %s", intent)

    engine = get_countvid_engine()

    if engine and perception.image is not None:
        image = perception.image
        h, w = image.shape[:2]

        # --- POINTBEAM: Crop to focus_roi if set ---
        roi_offset = (0, 0)  # (x_offset, y_offset) for coordinate translation
        if perception.focus_roi is not None:
            x1, y1, x2, y2 = perception.focus_roi
            # Ensure valid coordinates
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            image = image[y1:y2, x1:x2]
            roi_offset = (x1, y1)
            logger.info(
                "[countvid_executor_node] POINTBEAM: Cropped to ROI %s",
                perception.focus_roi,
            )

        # --- REFINEMENT: Adjust sensitivity if needed ---
        # Note: sensitivity_modifier would need to be passed to engine.count()
        # For now, we log it. CountGDEngine can be extended to accept threshold_shift.
        sensitivity = perception.sensitivity_modifier
        if sensitivity != 1.0:
            logger.info(
                "[countvid_executor_node] Sensitivity modifier: %.2f", sensitivity
            )

        # CountGD call
        count_val, raw_detections = engine.count(
            image, intent, target_size=(image.shape[1], image.shape[0])
        )

        # Translate detections back to original image coordinates if ROI was applied
        if roi_offset != (0, 0):
            x_off, y_off = roi_offset
            translated = []
            for det in raw_detections:
                x1, y1, x2, y2 = det
                translated.append([x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off])
            raw_detections = translated

        # --- FIX PYDANTIC: Convert list bboxes to dict format ---
        # PerceptionState.raw_detections expects List[Dict[str, Any]]
        intent_label = intent[0] if isinstance(intent, list) and intent else str(intent)
        formatted_detections = []
        for det in raw_detections:
            if isinstance(det, list) and len(det) >= 4:
                x1, y1, x2, y2 = det[:4]
                formatted_detections.append(
                    {
                        "bbox": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
                        "score": 1.0,
                        "label": intent_label,
                    }
                )
            elif isinstance(det, dict):
                formatted_detections.append(det)  # Already formatted

        updated_perception = perception.model_copy(
            update={
                "n_visible": count_val,
                "raw_detections": formatted_detections,
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

        # Use calibrated unit volume if available, else use prior
        unit_vol = (
            perception.estimated_unit_volume
            if perception.estimated_unit_volume > 0
            else ctx.unit_volume_prior
        )
        min_c, max_c = MathUtils.lattice_counting(total_volume, unit_vol)

        updates["n_volumetric_range"] = (min_c, max_c)
        updates["total_observed_volume"] = float(total_volume)
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

        # Normalize detections: CountGD returns list of [x1,y1,x2,y2],
        # but ReIDEngine + MathUtils expect {"bbox": {"x":..., "y":..., "w":..., "h":...}, "score": ...}
        normalized_detections = []
        if current_detections:
            for det in current_detections:
                if isinstance(det, list) and len(det) == 4:
                    # Convert [x1, y1, x2, y2] to {x, y, w, h} format
                    x1, y1, x2, y2 = det
                    normalized_detections.append(
                        {
                            "bbox": {
                                "x": float(x1),
                                "y": float(y1),
                                "w": float(x2 - x1),
                                "h": float(y2 - y1),
                            },
                            "score": 0.95,  # CountGD doesn't return scores, use high default
                            "features": None,
                        }
                    )
                elif isinstance(det, dict):
                    # Already in correct format (check if bbox needs conversion)
                    if "bbox" in det and isinstance(det["bbox"], list):
                        x1, y1, x2, y2 = det["bbox"]
                        det["bbox"] = {
                            "x": float(x1),
                            "y": float(y1),
                            "w": float(x2 - x1),
                            "h": float(y2 - y1),
                        }
                    normalized_detections.append(det)

        matched, new_dets = reid_engine.match_detections(
            normalized_detections, perception.current_frame_idx
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
              countvid_executor   sam2_depth         (v2e output)
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
    graph.add_node("countvid_executor_node", countvid_executor_node)
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
    graph.add_edge("vljepa_director_node", "countvid_executor_node")
    graph.add_edge("vljepa_director_node", "sam2_depth_node")

    # All paths converge at Fusion
    graph.add_edge("v2e_sensor_node", "fusion_engine_node")
    graph.add_edge("countvid_executor_node", "fusion_engine_node")
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

    logger.info("[build_recursive_graph] Graph compiled.")
    app = graph.compile()

    try:
        # Use .get_graph() on the compiled app
        app.get_graph().print_ascii()
    except Exception as e:
        logger.debug("Could not print graph: %s", e)

    return app


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
