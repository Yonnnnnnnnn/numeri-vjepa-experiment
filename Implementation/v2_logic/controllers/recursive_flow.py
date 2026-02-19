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
_dinov2_engine: Optional["DINOv2Engine"] = None
_density_predictor: Optional["DensityPredictor"] = None
_alphashape_wrapper: Optional["AlphaHullWrapper"] = None


def get_dinov2_engine():
    """Lazy-load DINOv2Engine."""
    global _dinov2_engine
    if _dinov2_engine is None:
        try:
            from ..models.dinov2_engine import DINOv2Engine

            _dinov2_engine = DINOv2Engine()
            logger.info("[Engines] DINOv2Engine initialized")
        except Exception as e:
            logger.warning("[Engines] Failed to load DINOv2Engine: %s", e)
    return _dinov2_engine


def get_density_predictor():
    """Lazy-load DensityPredictor."""
    global _density_predictor
    if _density_predictor is None:
        try:
            from ..models.density_predictor import DensityPredictor

            _density_predictor = DensityPredictor()
            logger.info("[Engines] DensityPredictor initialized")
        except Exception as e:
            logger.warning("[Engines] Failed to load DensityPredictor: %s", e)
    return _density_predictor


def get_alphashape_wrapper():
    """Lazy-load AlphaHullWrapper."""
    global _alphashape_wrapper
    if _alphashape_wrapper is None:
        try:
            from ..kernels.alphashape_wrapper import AlphaHullWrapper

            _alphashape_wrapper = AlphaHullWrapper()
            logger.info("[Engines] AlphaHullWrapper initialized")
        except Exception as e:
            logger.warning("[Engines] Failed to load AlphaHullWrapper: %s", e)
    return _alphashape_wrapper


def get_segmentation_engine():
    """Lazy-load SegmentationEngine (SAM2)."""
    global _segmentation_engine
    if _segmentation_engine is None:
        try:
            from ..models.segmentation_engine import SegmentationEngine

            # Loosened ROI margin from 0.05 to 0.02 to capture objects at the very edge of the frame
            _segmentation_engine = SegmentationEngine(roi_margin=0.02)
            logger.info(
                "[Engines] SegmentationEngine initialized with loosened ROI (0.02)"
            )
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

        return {
            "perception": {
                "v2e_spike_map": spike_map,
                "spike_energy": float(np.sum(spike_map)),
            }
        }

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
        # The engine instance's context (deque) is updated.
        _ = vjepa.encode(tensor)

        # PERSISTENCE FOR VISUALIZER: Save context to disk
        # This allows separate visualizer processes (like engine_v2) to access the latent state
        vjepa.export_context("vjepa_context_dump.pt")

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

        # --- DISCOVERY LOOP V3.3: Genesis Intent Matching ---
        # Replace hardcoded keywords with dynamic matching against genesis_intents
        genesis_labels = [
            gi.get("label", "").lower()
            for gi in perception.genesis_intents
            if not gi.get("is_contra", False)
        ]
        contra_labels = [
            ci.get("label", "").lower() for ci in perception.contra_intents
        ]

        discovered_labels = []
        new_contras = []

        # Parse hypothesis for mentioned objects
        for genesis_label in genesis_labels:
            # Check if any genesis intent label appears in the hypothesis
            if genesis_label in hypothesis and genesis_label not in [
                x.lower() for x in current_intent
            ]:
                discovered_labels.append(genesis_label)

        # Check for known Contra Intents mentioned (should stay suppressed)
        for contra_label in contra_labels:
            if contra_label in hypothesis:
                logger.info(
                    "[director] IMMUNITY: Known contra intent '%s' detected, ignoring",
                    contra_label,
                )

        # Check for NEW unknown objects not in genesis or contra lists
        # These need VLM classification (Step 12: is it New Genesis or Contra?)
        import re

        # Look for quoted object names or "found X" patterns
        mentioned_objects = re.findall(
            r'"([^"]+)"|found\s+(?:a|an)?\s*(\w+)', hypothesis
        )
        for match_groups in mentioned_objects:
            obj_name = (match_groups[0] or match_groups[1]).lower().strip()
            if not obj_name:
                continue
            is_known_genesis = any(obj_name in gl for gl in genesis_labels)
            is_known_contra = any(obj_name in cl for cl in contra_labels)
            is_current = obj_name in [x.lower() for x in current_intent]

            if not is_known_genesis and not is_known_contra and not is_current:
                # Step 12: Unknown anomaly → classify as Contra Intent
                # (VLM already analyzed in genesis; anything new is a distractor)
                new_contras.append(
                    {
                        "label": obj_name,
                        "latent_id": f"contra_{decision.loop_count}_{obj_name}",
                        "bbox": {},
                    }
                )
                logger.info(
                    "[director] STEP 12: New Contra Intent registered: '%s'",
                    obj_name,
                )

        if discovered_labels:
            for label in discovered_labels:
                if isinstance(label, str) and label not in current_intent:
                    current_intent.append(label)
            logger.info(
                "[director] DISCOVERY LOOP V3.3: Added genesis labels: %s",
                discovered_labels,
            )
            updates["active_intent"] = current_intent

        if new_contras:
            updated_contras = list(perception.contra_intents) + new_contras
            updates["contra_intents"] = updated_contras
            # Step 12.1: Generate negative masks for new contras
            # (Actual mask generation happens in fusion_engine_node)
            logger.info(
                "[director] STEP 12.1: %d new Contra Intents → Negative Masks pending",
                len(new_contras),
            )

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
                try:
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
                except Exception as e:
                    logger.warning("[director] Failed to parse SLM count: %s", e)

    # --- INITIAL VOLUME ESTIMATION FROM SLM ---
    # If we have an intent but no valid volume prior (still 0 or default 0.001), ask SLM.
    current_vol = perception.estimated_unit_volume
    if current_vol <= 0.001:  # Assuming 0.001 is the "uninformed" default
        # Get target label from updates (if just discovered) or existing state
        target_list = updates.get(
            "active_intent", perception.active_intent or list(ctx.main_intent)
        )
        target_label = target_list[0] if target_list else "object"

        slm = get_slm_engine()
        if slm:
            try:
                slm_vol = slm.estimate_object_volume(target_label)
                if slm_vol > 0 and slm_vol != 0.001:
                    updates["estimated_unit_volume"] = slm_vol
                    logger.info(
                        "[director] SLM PHYSICAL PRIOR: Estimated volume for '%s' is %.5f m^3",
                        target_label,
                        slm_vol,
                    )
            except Exception as e:
                logger.warning("[director] Failed to get SLM volume prior: %s", e)

    # Always ensure active_intent is populated (even if no changes)
    if not perception.active_intent and "active_intent" not in updates:
        updates["active_intent"] = list(ctx.main_intent)

    # V3.3.4: Reset Circuit Breaker at start of sensor orbit
    updates["is_volumetric_data_fresh"] = False

    if updates:
        return {"perception": updates}

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

        return {
            "perception": {
                "n_visible": count_val,
                "raw_detections": formatted_detections,
            }
        }

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

    # 2. Segmentation (V3.1 Union Masking)
    masks = []
    clusters = []
    if seg_engine:
        seg_res = seg_engine.segment_frame(image)
        # Apply Union Masking from Phase 4
        merged_res = seg_engine.merge_overlapping_masks(seg_res)
        masks = merged_res.masks
        updates["masks"] = masks

        # 3. Detect volumetric clusters for AlphaHull
        if depth_map is not None:
            clusters = seg_engine.detect_volumetric_clusters(merged_res, depth_map)

            # --- VOLUME AGGREGATION FIX ---
            from ..utils.math_utils import MathUtils

            # Scale depth map to real-world units (meters)
            # depth_map is normalized 0-1, scale_factor (e.g. 1.0m) gives physical Z
            scale_factor = ctx.depth_scale_factor

            # Sanity check: if scale factor is clearly in cm (>10), convert to meters
            if scale_factor > 10.0:
                logger.warning(
                    "[sam2_depth_node] Depth scale factor %.1f likely in cm, converting to meters",
                    scale_factor,
                )
                scale_factor /= 100.0

            # Normalize depth map if it's 8-bit (0-255) to 0-1 range before scaling
            if np.max(depth_map) > 2.0:  # Assuming anything > 2.0 is not meters
                logger.warning(
                    "[sam2_depth_node] Depth map values > 2.0 (max=%.1f), normalizing to 0-1",
                    np.max(depth_map),
                )
                scaled_depth = (depth_map / 255.0) * scale_factor
            else:
                scaled_depth = depth_map * scale_factor

            total_v = 0.0
            # Use camera intrinsics if available inside ctx, else default
            # Note: ctx is GlobalContext (Pydantic), so use getattr or direct access
            intrinsics = getattr(ctx, "camera_intrinsics", {}) or {}
            fx = intrinsics.get("fx", 500.0)
            fy = intrinsics.get("fy", 500.0)

            for cluster in clusters:
                mask = cluster["combined_mask"]
                # 3.1 Project to Physical Volume
                cluster_v, cluster_pts = MathUtils.project_to_physical_volume(
                    depth_map=scaled_depth,
                    mask=mask,
                    fx=fx,
                    fy=fy,
                )

                # 3.2 Extract 2D Bounding Box for Spatial Branding (Phase 4)
                y_coords, x_coords = np.where(mask > 0)
                if len(x_coords) > 0:
                    x1, y1 = np.min(x_coords), np.min(y_coords)
                    x2, y2 = np.max(x_coords), np.max(y_coords)
                    cluster["bbox"] = {
                        "x": float(x1),
                        "y": float(y1),
                        "w": float(x2 - x1),
                        "h": float(y2 - y1),
                    }
                else:
                    cluster["bbox"] = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}

                cluster["volume_m3"] = cluster_v
                cluster["points"] = cluster_pts  # Store real 3D points for V3 Math
                total_v += cluster_v

            # Store in cubic meters (standard unit)
            # No longer multiplying by 1e6 to avoid scale confusion
            updates["total_observed_volume"] = total_v
            logger.info(
                "[sam2_depth_node] Aggregated volume: %.6f m^3 (%.2f cm^3) from %d clusters",
                total_v,
                total_v * 1e6,
                len(clusters),
            )

    # Store clusters for later V3 Math processing
    updates["point_cloud_summary"] = {"clusters": clusters}

    return {"perception": updates}


def density_sensing_node(state: RecursiveFlowState) -> Dict[str, Any]:
    """
    Sovereignty 3: Analyze texture turbidity to predict rho (density).
    Uses DINOv2 specularity analysis and MLP Regressor.
    """
    logger.info("[density_sensing_node] Analyzing physical density")
    perception = state["perception"]
    dinov2 = get_dinov2_engine()
    mlp = get_density_predictor()

    if perception.image is not None and dinov2 and mlp:
        # 1. Extract texture features
        specularity = dinov2.analyze_specularity(perception.image)

        # 2. Predict rho using MLP
        # For simplicity, we use binary features or a heuristic if data is thin
        features = np.array([[specularity]])
        rho = mlp.predict(features)[0]

        return {"perception": {"rho": float(rho)}}

    return {}


def v3_math_node(state: RecursiveFlowState) -> Dict[str, Any]:
    """
    Arithmetic Kernel: Compute final N_vol using Golden Alpha and Reconciliation.
    Formula: N_vol = (V_stack * rho) / V_unit
    """
    logger.info("[v3_math_node] Computing volumetric reconciliation")
    perception = state["perception"]
    ctx = state["ctx"]
    alpha_wrapper = get_alphashape_wrapper()
    from ..utils.math_utils import MathUtils

    if not alpha_wrapper or not perception.point_cloud_summary:
        return {}

    clusters = perception.point_cloud_summary.get("clusters", [])
    if not clusters:
        return {}

    # 1. Intent-Based Cluster Filtering (Reconciliation Layer)
    # Get all target SKUs from Intent Genesis (V3.3 logic)
    target_intents = []
    if perception.genesis_intents:
        target_intents = [
            i["label"].lower()
            for i in perception.genesis_intents
            if not i.get("is_contra", False)
        ]

    # Fallback to main intent if genesis is empty
    if not target_intents:
        target_intents = (
            ctx.main_intent
            if isinstance(ctx.main_intent, list)
            else [ctx.main_intent or "items"]
        )

    raw_detections = perception.raw_detections or []
    filtered_clusters = []
    v_stack = 0.0

    logger.info("[v3_math_node] Target intents for reconciliation: %s", target_intents)

    for cluster in clusters:
        cluster_bbox = cluster.get("bbox")
        if not cluster_bbox or cluster_bbox["w"] == 0:
            continue

        # Check if this cluster overlaps with any detection matching our target intent
        is_target = False
        for det in raw_detections:
            det_label = det.get("label", "").lower()

            # Semantic Check: Does the detection label relate to any of our target intents?
            matches_intent = any(
                intent.lower() in det_label or det_label in intent.lower()
                for intent in target_intents
            )

            if matches_intent:
                # Calculate IoU - Reduced threshold (0.1) because SAM2 masks and DINO boxes differ in shape
                iou = MathUtils.calculate_bbox_overlap(cluster_bbox, det["bbox"])
                if iou > 0.1:
                    is_target = True
                    break

        if is_target:
            filtered_clusters.append(cluster)
            v_stack += cluster.get("volume_m3", 0.0)
        else:
            # OPTIMIZATION: If we only have target intents defined, be more aggressive in keeping clusters
            # if they have ANY overlap with visual detections.
            for det in raw_detections:
                iou = MathUtils.calculate_bbox_overlap(cluster_bbox, det["bbox"])
                if iou > 0.05:
                    v_stack += cluster.get("volume_m3", 0.0)
                    is_target = True
                    break

            if not is_target:
                logger.info(
                    "[v3_math_node] Excluding cluster with mean_depth %.2f (No overlapping target intent)",
                    cluster.get("mean_depth", 0.0),
                )

    # 2. V3 Core Calibration
    target_unit_v = (
        perception.estimated_unit_volume
        if perception.estimated_unit_volume > 0
        else ctx.unit_volume_prior
    )

    updates = {}

    # 2. Step 3.5: Golden Alpha Calibration
    # If we see exactly 1 object (and no Golden Alpha set), we calibrate.
    if perception.n_visible == 1 and perception.golden_alpha == 0:
        logger.info("[v3_math_node] Step 3.5: Calibrating Golden Alpha")
        try:
            # Use real points from the cluster if available
            calibration_points = None
            if clusters and "points" in clusters[0]:
                calibration_points = clusters[0]["points"]
            else:
                # Fallback to simple simulated points only if no cluster data
                calibration_points = np.random.rand(100, 3)

            golden_alpha = alpha_wrapper.find_golden_alpha(
                calibration_points, target_unit_v
            )
            updates["golden_alpha"] = float(golden_alpha)
            logger.info(
                "[v3_math_node] Golden Alpha calibrated: %.4f using %d points",
                golden_alpha,
                len(calibration_points),
            )
        except Exception as e:
            logger.warning("[v3_math_node] Calibration failed: %s", e)

    # 3. Step 8-9: Final Volumetric Count (V3.3 Mixed-Object Logic)
    # Formula: N_vol = sum(V_cluster / V_unit_for_label)
    n_vol = 0.0
    slm = get_slm_engine()  # To fetch priors if not cached

    for cluster in filtered_clusters:
        c_vol = cluster.get("volume_m3", 0.0)

        # Determine the best label for this cluster
        best_label = "item"
        best_iou = 0.0
        for det in raw_detections:
            iou = MathUtils.calculate_bbox_overlap(
                cluster.get("bbox", {}), det.get("bbox", {})
            )
            if iou > best_iou:
                best_iou = iou
                best_label = det.get("label", "item")

        # Get specific unit volume for this label
        # In a production run, we'd cache these in PerceptionState
        u_v = slm.estimate_object_volume(best_label) if slm else target_unit_v

        # Calculate fractional count
        cluster_n = (c_vol * perception.rho) / u_v if u_v > 1e-9 else 0.0
        n_vol += cluster_n

        logger.info(
            "[v3_math_node] Cluster (%s): V=%.6f m^3 -> n=%.2f (using v_unit=%.6f)",
            best_label,
            c_vol,
            cluster_n,
            u_v,
        )

    logger.info(
        "[v3_math_node] 3DC RECONCILIATION: Total n_vol = %.2f across %d filtered clusters",
        n_vol,
        len(filtered_clusters),
    )

    # Phase 6: Sanity Guard Check
    from ..utils.math_utils import SanityGuard

    is_valid, reason = SanityGuard.validate_volumetric_bounds(n_vol)
    if not is_valid:
        logger.warning(
            "[v3_math_node] Sanity Check Failed: %s. Clipping n_vol.", reason
        )
        n_vol = max(0.0, min(n_vol if not np.isnan(n_vol) else 0.0, 1000.0))

    # Update n_volumetric_range for the Logic Gate
    # We use a small epsilon for the range
    updates["n_volumetric_range"] = (int(n_vol), int(np.ceil(n_vol)))
    updates["total_observed_volume"] = v_stack  # Persist V_stack
    updates["is_volumetric_data_fresh"] = True  # Step 3.6: Circuit Breaker Reset

    return {"perception": updates}


def fusion_engine_node(state: RecursiveFlowState) -> Dict[str, Any]:
    """
    Fuse spike data with SAM2 masks and TRACK objects.
    Implements Re-ID to maintain consistent counts across loops.
    """
    logger.info("[fusion_engine_node] Fusing sensor data & Tracking")

    fusion_engine = get_fusion_engine()
    reid_engine = get_reid_engine()
    perception = state["perception"]
    ctx = state["ctx"]

    updates = {}

    # 1. Fusion (Multi-Shield v3.1)
    if fusion_engine and perception.v2e_spike_map is not None:
        masks = perception.masks if perception.masks else []

        # Run Multi-Shield Fusion from Phase 4
        # We need to provide track similarities if ReID just ran,
        # but ReID runs AFTER fusion in the old node.
        # V3.1: We'll run ReID first or pass latent similarities.

        result = fusion_engine.fuse_spike_mask(
            spike_map=perception.v2e_spike_map,
            masks=masks,
            n_visible=perception.n_visible,
            n_volumetric_range=perception.n_volumetric_range,
        )

        n_vol_range = perception.n_volumetric_range
        # Orchestrate shields
        result = fusion_engine.run_shields(
            fusion_result=result,
            n_visible=perception.n_visible,
            n_volumetric_range=n_vol_range,
            v_stack=perception.total_observed_volume,
            v_unit=ctx.unit_volume_prior,  # Fallback to prior
        )

        updates["residual_spike_energy"] = result.residual_spike_energy
        updates["unexplained_blobs"] = result.unexplained_blobs
        updates["shield_scores"] = result.shield_scores
        updates["fusion_confidence"] = result.fusion_confidence

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

    return {"perception": updates}


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

        # Determine anomalies dynamically (V3.1 Triangulation)
        gate_decision = engine.evaluate(
            n_visible=perception.n_visible,
            n_volumetric_range=perception.n_volumetric_range,
            residue_ratio=residue_ratio,
            unexplained_blob_area=0.0,
            detection_confidence=perception.fusion_confidence,  # Use Multi-Shield confidence
            current_loop_count=decision.loop_count,
            has_spatial_anomaly=perception.shield_scores.get("spatial", 1.0) < 0.6,
            has_volumetric_anomaly=perception.shield_scores.get("volumetric", 1.0)
            < 0.6,
        )

        # --- STEP 3.6: CIRCUIT BREAKER (Stale Data Guard) ---
        if (
            gate_decision.anomaly_type.value == "volumetric"
            and not perception.is_volumetric_data_fresh
        ):
            logger.warning(
                "[logic_gate_node] CIRCUIT BREAKER: Volumetric anomaly with STALE data detected. Forcing Exit."
            )
            return {
                "decision": {
                    "status": "exit",
                    "anomaly_type": "none",
                    "logic_gate_result": {
                        "rule_applied": "CIRCUIT_BREAKER_STALE_DATA",
                        "confidence": 0.0,
                        "action": "exit",
                        "reasoning": "Circuit Breaker: Volumetric data is stale (V3 Math did not update). Terminating to prevent loop deadlock.",
                    },
                }
            }

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

        return {"decision": new_decision.model_dump(exclude_unset=True)}

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

        return {
            "decision": {
                "slm_triggered": True,
                "slm_reasoning": result.reasoning_text,
                "slm_hypothesis": result.hypothesis,
            }
        }

    return {}


def route_to_genesis_or_brain(
    state: RecursiveFlowState,
) -> Literal["intent_genesis_node", "vjepa_brain_node"]:
    """
    Conditional edge: Route to Intent Genesis (Step 0) ONLY if not initialized.
    """
    decision = state["decision"]
    if not decision.is_genesis_complete:
        logger.info("[route] Genesis not complete, routing to Step 0")
        return "intent_genesis_node"

    return "vjepa_brain_node"


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

    return {
        "decision": {
            "loop_count": decision.loop_count + 1,
            "status": "looping",
        }
    }


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
      │
      v
    intent_genesis_node (Step 0: Scout + Analyst)
      │
      v
    vjepa_brain_node (Step 1: Anchor)
      │
      v
    vljepa_director_node (Step 2: Director)
      ├───────────────┬───────────────┬─────────────┐
      v               v               v             v
    countvid     sam2_depth     density_sense    v2e_sensor
      │               │               │             │
      └───────────────┴───────────────┴─────────────┘
                              │
                              v
                       v3_math_node (Steps 3.5-9)
                              │
                              v
                     fusion_engine_node (Step 10)
                              │
                              v
                       logic_gate_node (Step 11)
                              │
                 ┌────────────┴────────────┐
                 v                         v
               (exit)              targeted_slm_node (Step 12)
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
    # V3.3: Import Intent Genesis (Step 0)
    from .intent_genesis_node import intent_genesis_node

    graph.add_node("intent_genesis_node", intent_genesis_node)
    graph.add_node("v2e_sensor_node", v2e_sensor_node)
    graph.add_node("vjepa_brain_node", vjepa_brain_node)
    graph.add_node("vljepa_director_node", vljepa_director_node)
    graph.add_node("countvid_executor_node", countvid_executor_node)
    graph.add_node("sam2_depth_node", sam2_depth_node)
    graph.add_node("density_sensing_node", density_sensing_node)
    graph.add_node("v3_math_node", v3_math_node)
    graph.add_node("fusion_engine_node", fusion_engine_node)
    graph.add_node("logic_gate_node", logic_gate_node)
    graph.add_node("targeted_slm_node", targeted_slm_node)
    graph.add_node("interpolation_node", interpolation_node)

    # --- Add Edges (V3.3 Sovereignty Chain) ---

    # V3.3.4: START → check initialization → Genesis (Step 0) OR Brain (Step 1)
    graph.add_conditional_edges(
        START,
        route_to_genesis_or_brain,
        {
            "intent_genesis_node": "intent_genesis_node",
            "vjepa_brain_node": "vjepa_brain_node",
        },
    )

    # Genesis flows to Brain once complete
    graph.add_edge("intent_genesis_node", "vjepa_brain_node")

    # Brain → Director (Step 2)
    graph.add_edge("vjepa_brain_node", "vljepa_director_node")

    # Director → Parallel Senses (Sovereignties 1, 2, 3)
    graph.add_edge("vljepa_director_node", "v2e_sensor_node")
    graph.add_edge("vljepa_director_node", "sam2_depth_node")
    graph.add_edge("vljepa_director_node", "countvid_executor_node")
    graph.add_edge("vljepa_director_node", "density_sensing_node")

    # Senses converge at Math Kernel
    graph.add_edge("v2e_sensor_node", "v3_math_node")
    graph.add_edge("sam2_depth_node", "v3_math_node")
    graph.add_edge("countvid_executor_node", "v3_math_node")
    graph.add_edge("density_sensing_node", "v3_math_node")

    # Math → Fusion (Validation)
    graph.add_edge("v3_math_node", "fusion_engine_node")

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
