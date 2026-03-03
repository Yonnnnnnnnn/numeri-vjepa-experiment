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
  │  <DirectorNodes>   → latent_director_node                                 │
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

Terminals       : "v2e_sensor", "vjepa_brain", "latent_director", etc.

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

from ..models.lnn_knowledge_base import get_lnn_kb

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
            _density_predictor.calibrate_heuristic()
            logger.info("[Engines] DensityPredictor initialized and calibrated")
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

    V5.0: Also accumulates Causal Focus Score per tracked latent ID
    and maintains a rolling frame buffer for Back-in-Time retrieval.
    
    V5.1: Composite Focus Score = (Centrality * 0.4) + (Scale Growth * 0.4) + (Velocity Alignment * 0.2)
          Plus User Pointing Causal Indicator (hand-object intersection boost).
    """
    logger.info("[vjepa_brain_node] Encoding latent context")
    vjepa = get_vjepa_engine()
    perception = state["perception"]
    ctx = state["ctx"]

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

        # =====================================================================
        # V5.1: COMPOSITE FOCUS SCORE WITH VELOCITY, SCALE, AND POINTING
        # =====================================================================
        updates = {}
        focus_scores = dict(perception.latent_focus_scores)  # Copy current
        decay_factor = 0.85  # Slightly higher decay for smoother accumulation
        
        # Build bbox history from causal buffer for velocity/scale calculation
        buffer = list(perception.causal_frame_buffer)
        max_buffer = ctx.causal_buffer_frames
        
        # Track bbox history per latent_id: {latent_id: [(frame_idx, bbox, area), ...]}
        bbox_history = {}
        for gi in perception.genesis_intents:
            latent_id = gi.get("label", "")
            bbox = gi.get("bbox")
            if latent_id and bbox:
                if latent_id not in bbox_history:
                    bbox_history[latent_id] = []
                # Add current frame bbox
                area = bbox.get("w", 0) * bbox.get("h", 0)
                bbox_history[latent_id].append((perception.current_frame_idx, bbox, area))
        
        # Extract historical bboxes from buffer
        for entry in buffer:
            frame_idx = entry.get("frame_idx", 0)
            bbox_snap = entry.get("bbox_snapshot", {})
            for lid, bb in bbox_snap.items():
                if lid and bb:
                    if lid not in bbox_history:
                        bbox_history[lid] = []
                    area = bb.get("w", 0) * bb.get("h", 0)
                    bbox_history[lid].append((frame_idx, bb, area))
        
        # Detect "Human Hand" from contra_intents for pointing causal indicator
        hand_contras = [
            ci for ci in perception.contra_intents 
            if "hand" in ci.get("label", "").lower()
        ]
        
        for intent in perception.genesis_intents:
            latent_id = intent.get("label", "")
            bbox = intent.get("bbox")
            if not latent_id or not bbox:
                continue

            # Calculate centrality (re-use genesis logic)
            from .intent_genesis_node import _calculate_centrality_score
            centrality = _calculate_centrality_score(bbox)
            
            # --- V5.1: SCALE GROWTH CALCULATION ---
            # Object moving closer should have increasing area
            scale_growth = 0.0
            history = bbox_history.get(latent_id, [])
            if len(history) >= 2:
                # Compare current area with average of previous 2-3 frames
                current_area = history[-1][2]
                prev_areas = [h[2] for h in history[:-1] if h[2] > 0]
                if prev_areas:
                    avg_prev_area = sum(prev_areas) / len(prev_areas)
                    if avg_prev_area > 0:
                        scale_ratio = current_area / avg_prev_area
                        # Boost if growing (ratio > 1.0), penalize if shrinking
                        scale_growth = max(0.0, min(2.0, (scale_ratio - 1.0) * 2 + 1.0))
                        # Normalize: ratio=1.0 -> 1.0, ratio=1.05 -> 1.1, ratio=0.95 -> 0.9
            
            # --- V5.1: VELOCITY ALIGNMENT CALCULATION ---
            # Check if object is moving toward center
            velocity_alignment = 1.0  # Default neutral
            if len(history) >= 2:
                curr_bbox = history[-1][1]
                prev_bbox = history[-2][1]
                
                curr_cx = curr_bbox.get("x", 0) + curr_bbox.get("w", 0) / 2
                curr_cy = curr_bbox.get("y", 0) + curr_bbox.get("h", 0) / 2
                prev_cx = prev_bbox.get("x", 0) + prev_bbox.get("w", 0) / 2
                prev_cy = prev_bbox.get("y", 0) + prev_bbox.get("h", 0) / 2
                
                # Velocity vector
                vel_x = curr_cx - prev_cx
                vel_y = curr_cy - prev_cy
                
                # Vector to center (0.5, 0.5)
                center_x, center_y = 0.5, 0.5
                to_center_x = center_x - curr_cx
                to_center_y = center_y - curr_cy
                
                # Dot product: positive if moving toward center
                dot_product = vel_x * to_center_x + vel_y * to_center_y
                if dot_product > 0:
                    velocity_alignment = 1.0 + min(1.0, abs(dot_product) * 5)  # Boost up to 2.0
                elif dot_product < 0:
                    velocity_alignment = max(0.5, 1.0 - abs(dot_product) * 2)  # Penalize down to 0.5
            
            # --- V5.1: USER POINTING CAUSAL INDICATOR ---
            # Check if hand is pointing at or intersecting this object
            pointing_boost = 0.0
            if hand_contras and bbox:
                obj_x1 = bbox.get("x", 0)
                obj_y1 = bbox.get("y", 0)
                obj_x2 = obj_x1 + bbox.get("w", 0)
                obj_y2 = obj_y1 + bbox.get("h", 0)
                
                for hand in hand_contras:
                    hand_bbox = hand.get("bbox", {})
                    if not hand_bbox:
                        continue
                    
                    hand_x1 = hand_bbox.get("x", 0)
                    hand_y1 = hand_bbox.get("y", 0)
                    hand_x2 = hand_x1 + hand_bbox.get("w", 0)
                    hand_y2 = hand_y1 + hand_bbox.get("h", 0)
                    
                    # Check for intersection or proximity
                    # Simple AABB intersection check
                    intersects = not (obj_x2 < hand_x1 or obj_x1 > hand_x2 or 
                                     obj_y2 < hand_y1 or obj_y1 > hand_y2)
                    
                    if intersects:
                        pointing_boost = 2.0  # Massive boost for hand-object interaction
                        logger.info(
                            "[PointingCausal] Hand detected intersecting with '%s'!",
                            latent_id
                        )
                        break
            
            # --- COMPOSITE FOCUS SCORE ---
            # Formula: [(Centrality * 0.4) + (Scale Growth * 0.4) + (Velocity Alignment * 0.2)] + Pointing Boost
            base_score = (centrality * 0.4) + (scale_growth * 0.4) + (velocity_alignment * 0.2)
            composite_score = base_score + pointing_boost
            
            # Exponential moving average accumulation
            old_score = focus_scores.get(latent_id, 0.0)
            new_score = old_score * decay_factor + composite_score
            focus_scores[latent_id] = round(new_score, 4)
            
            # Log detailed breakdown for debugging
            if pointing_boost > 0 or scale_growth > 1.1 or velocity_alignment > 1.1:
                logger.info(
                    "[vjepa_brain_node] V5.1 Focus '%s': centrality=%.2f, scale=%.2f, vel=%.2f, pointing=%.2f => composite=%.2f, accumulated=%.2f",
                    latent_id, centrality, scale_growth, velocity_alignment, pointing_boost, composite_score, new_score
                )

        if focus_scores:
            updates["latent_focus_scores"] = focus_scores
            logger.info(
                "[vjepa_brain_node] V5.1 Focus Scores: %s",
                {k: f"{v:.2f}" for k, v in focus_scores.items()},
            )

        # --- Causal Frame Buffer (Rolling Window) ---
        buffer_entry = {
            "frame_idx": perception.current_frame_idx,
            "image": perception.image,
            "latent_confidences": dict(focus_scores),
            # Store bbox snapshot for history tracking
            "bbox_snapshot": {
                gi.get("label"): gi.get("bbox") 
                for gi in perception.genesis_intents 
                if gi.get("label") and gi.get("bbox")
            },
        }
        buffer.append(buffer_entry)

        # Trim to max buffer size (FIFO)
        if len(buffer) > max_buffer:
            buffer = buffer[-max_buffer:]

        updates["causal_frame_buffer"] = buffer

        if updates:
            return {"perception": updates}

        return {}

    return {}


# ... (Node Definitions until Fusion)


def latent_director_node(state: RecursiveFlowState) -> Dict[str, Any]:
    """
    Generate or update intent list based on V-JEPA latent.
    Acts as the "Sutradara" (Director) of the system.

    Implements:
    - V5.0 Wait-and-Watch: Focal trigger based on accumulated centrality dwell
    - Discovery Loop: Parse SLM hypothesis for new object labels
    - Refinement Loop: Adjust sensitivity or set PointBeam ROI
    - V5.1 LNN Last Gate: Final confirmation for all VLM labels
    """
    from .recursive_flow import get_slm_engine, get_vjepa_engine, get_lnn_kb

    slm = get_slm_engine()
    vjepa = get_vjepa_engine()
    lnn = get_lnn_kb()

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

    # =========================================================================
    # V5.0/5.1: WAIT-AND-WATCH FOCAL TRIGGER + LNN GATE
    # =========================================================================
    focus_scores = perception.latent_focus_scores
    threshold = ctx.focal_trigger_threshold
    buffer = perception.causal_frame_buffer

    # Build a lookup map: label -> bbox for fast retrieval
    label_to_bbox = {
        gi.get("label"): gi.get("bbox")
        for gi in perception.genesis_intents
        if gi.get("label") and gi.get("bbox")
    }
    logger.info(
        "[FocalTrigger] Available genesis labels: %s",
        list(label_to_bbox.keys()),
    )

    for latent_id, score in focus_scores.items():
        # Trigger focal analysis if score reached threshold
        logger.info(
            "[FocalTrigger] Score Check: '%s' score=%.2f vs threshold=%.2f → %s",
            latent_id, score, threshold,
            "TRIGGERED" if score >= threshold else "below"
        )
        if score >= threshold and latent_id not in current_intent:
            logger.info(
                "[FocalTrigger] V5.0 WAIT-AND-WATCH: Latent '%s' reached "
                "focus score %.2f (threshold=%.1f). Triggering focal analysis.",
                latent_id,
                score,
                threshold,
            )

            # --- LATENT ANTICIPATION: Back-in-Time Retrieval ---
            best_frame = None
            best_confidence = -1.0
            for entry in buffer:
                entry_conf = entry.get("latent_confidences", {}).get(latent_id, 0.0)
                if entry_conf > best_confidence:
                    best_confidence = entry_conf
                    best_frame = entry.get("image")

            if best_frame is not None and slm:
                try:
                    # Find the bbox for this latent from genesis_intents
                    # Use the pre-built lookup map for reliability
                    target_bbox = label_to_bbox.get(latent_id)
                    
                    if target_bbox is None:
                        logger.warning(
                            "[FocalTrigger] No bbox found for latent_id '%s'. Available: %s",
                            latent_id,
                            list(label_to_bbox.keys()),
                        )
                        continue

                    logger.info(
                        "[FocalTrigger] Found bbox for '%s': %s",
                        latent_id,
                        target_bbox,
                    )

                    focal_label = slm.generate_focal_intent(
                        frame=best_frame, roi=target_bbox, latent_id=latent_id
                    )

                    if focal_label:
                        # --- V5.1: LNN AS LAST GATE (GEBAG TERAKHIR) ---
                        lnn_confidence = lnn.validate_intent(focal_label)
                        
                        # --- V5.1: PROVISIONAL ACCEPTANCE PATH ---
                        # If LNN rejects (score 0.0) but focus score is very high,
                        # it might be a novel product not in LNN knowledge base.
                        # Accept if: (1) LNN >= 0.7, OR (2) Focus Score >= 2.0 AND VLM triggered
                        high_focus_threshold = 2.0  # Very high accumulated focus
                        is_high_focus = score >= high_focus_threshold
                        
                        if lnn_confidence >= 0.7:
                            # Standard LNN confirmation
                            if focal_label not in current_intent:
                                current_intent.append(focal_label)
                            updates["active_intent"] = current_intent
                            logger.info(
                                "[FocalTrigger] LNN CONFIRMED: '%s' (conf=%.2f) promoted to active_intent",
                                focal_label,
                                lnn_confidence,
                            )
                        elif is_high_focus and lnn_confidence > 0.0:
                            # Borderline case: LNN uncertain but focus is high
                            logger.info(
                                "[FocalTrigger] LNN UNCERTAIN but HIGH FOCUS: '%s' (lnn=%.2f, focus=%.2f). Provisional acceptance.",
                                focal_label, lnn_confidence, score
                            )
                            if focal_label not in current_intent:
                                current_intent.append(focal_label)
                            updates["active_intent"] = current_intent
                        elif is_high_focus and lnn_confidence == 0.0:
                            # V5.1: LNN rejects but focus is extremely high → Provisional Acceptance
                            logger.warning(
                                "[FocalTrigger] LNN REJECTED but EXTREME FOCUS: '%s' (lnn=%.2f, focus=%.2f). "
                                "PROVISIONAL ACCEPTANCE as novel product (not in LNN KB).",
                                focal_label, lnn_confidence, score
                            )
                            # Mark as provisional with metadata
                            if focal_label not in current_intent:
                                current_intent.append(focal_label)
                            updates["active_intent"] = current_intent
                            
                            # Track provisional accepts for later review
                            if "provisional_accepts" not in updates:
                                updates["provisional_accepts"] = []
                            updates["provisional_accepts"].append({
                                "label": focal_label,
                                "latent_id": latent_id,
                                "focus_score": score,
                                "lnn_score": lnn_confidence,
                                "frame_idx": perception.current_frame_idx,
                            })
                        else:
                            # LNN rejects with low focus → Likely noise/distractor
                            logger.info(
                                "[FocalTrigger] LNN REJECTED: '%s' (conf=%.2f, focus=%.2f). Registering as Contra Intent.",
                                focal_label, lnn_confidence, score
                            )
                            # Register as contra to suppress in future frames
                            updated_contras = list(perception.contra_intents)
                            updated_contras.append(
                                {
                                    "label": focal_label,
                                    "latent_id": f"contra_focal_{latent_id}",
                                    "bbox": target_bbox,
                                }
                            )
                            updates["contra_intents"] = updated_contras
                except Exception as exc:
                    logger.error("[FocalTrigger] Focal analysis failed: %s", exc, exc_info=True)

    # Phase 9: Adaptive Intent Update from SLM Hypothesis
    if decision.slm_hypothesis:
        hypothesis = decision.slm_hypothesis.lower()
        logger.info("[director] SLM Hypothesis received: %s", decision.slm_hypothesis)

        # --- DISCOVERY LOOP V5.1: LNN Enhanced Step 12 ---
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
            if genesis_label in hypothesis and genesis_label not in [
                x.lower() for x in current_intent
            ]:
                discovered_labels.append(genesis_label)

        # Step 12: Handle NEW unknown objects (LNN Gate)
        import re

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
                # --- V5.1: LNN deciding Step 12 logic ---
                lnn_conf = lnn.validate_intent(obj_name)
                
                # --- V5.1: PROVISIONAL ACCEPTANCE FOR STEP 12 ---
                # Check if this object has high focus score (already being tracked)
                obj_focus_score = focus_scores.get(obj_name, 0.0)
                is_high_focus = obj_focus_score >= 2.0
                
                if lnn_conf >= 0.7:
                    # It's a valid product we missed! Promote to genesis
                    logger.info(
                        "[director] STEP 12+: LNN validated misses product '%s' (conf=%.2f). Promoting to active_intent.",
                        obj_name, lnn_conf
                    )
                    discovered_labels.append(obj_name)
                elif is_high_focus and lnn_conf > 0.0:
                    # Borderline: LNN uncertain but focus is high
                    logger.info(
                        "[director] STEP 12: LNN uncertain '%s' (conf=%.2f) but HIGH FOCUS (%.2f). Provisional acceptance.",
                        obj_name, lnn_conf, obj_focus_score
                    )
                    discovered_labels.append(obj_name)
                elif is_high_focus and lnn_conf == 0.0:
                    # V5.1: Novel product not in LNN KB but high focus
                    logger.warning(
                        "[director] STEP 12: LNN rejected '%s' (conf=0.0) but EXTREME FOCUS (%.2f). "
                        "PROVISIONAL ACCEPTANCE as novel product.",
                        obj_name, obj_focus_score
                    )
                    discovered_labels.append(obj_name)
                else:
                    # It's noise/distractor → Contra Intent
                    new_contras.append(
                        {
                            "label": obj_name,
                            "latent_id": f"contra_{decision.loop_count}_{obj_name}",
                            "bbox": {},
                        }
                    )
                    logger.info(
                        "[director] STEP 12: LNN rejected noise '%s' (conf=%.2f, focus=%.2f). Registering as Contra Intent.",
                        obj_name, lnn_conf, obj_focus_score
                    )

        if discovered_labels:
            for label in discovered_labels:
                if label not in [x.lower() for x in current_intent]:
                    # Find original casing if possible
                    orig_label = next(
                        (
                            gi.get("label")
                            for gi in perception.genesis_intents
                            if gi.get("label", "").lower() == label
                        ),
                        label,
                    )
                    current_intent.append(orig_label)

            updates["active_intent"] = current_intent
            logger.info("[director] Discovery Update: %s", discovered_labels)

            # Activate Latent Anchors
            new_anchors = []
            for gi in perception.genesis_intents:
                if (
                    gi.get("label", "").lower() in discovered_labels
                    and gi.get("latent_pose") is not None
                ):
                    new_anchors.append(gi["latent_pose"])

            if new_anchors:
                current_anchors = list(perception.active_latent_anchors)
                current_anchors.extend(new_anchors)
                updates["active_latent_anchors"] = current_anchors

        if new_contras:
            updated_contras = list(perception.contra_intents) + new_contras
            updates["contra_intents"] = updated_contras

        # REFINEMENT LOOP (Sensitivity + PointBeam Focus)
        refinement_hints = ["occluded", "hidden", "behind", "blocked", "covered"]
        if any(hint in hypothesis for hint in refinement_hints):
            new_sensitivity = min(sensitivity * 1.2, 2.0)
            updates["sensitivity_modifier"] = new_sensitivity
            logger.info(
                "[director] Refinement Loop: Sensitivity adjusted to %.2f",
                new_sensitivity,
            )

        if "corner" in hypothesis or "edge" in hypothesis:
            h, w = (
                perception.image.shape[:2]
                if perception.image is not None
                else (480, 640)
            )
            focus_roi = (w // 2, h // 2, w, h)
            updates["focus_roi"] = focus_roi
            logger.info(
                "[PointBeamFocus] REFINEMENT: Shifting attention to Bottom-Right %s",
                focus_roi,
            )
        elif "left" in hypothesis:
            h, w = (
                perception.image.shape[:2]
                if perception.image is not None
                else (480, 640)
            )
            focus_roi = (0, 0, w // 2, h)
            updates["focus_roi"] = focus_roi
            logger.info(
                "[PointBeamFocus] REFINEMENT: Shifting attention to LEFT-HALF %s",
                focus_roi,
            )

        # --- FIX 3: SLM ANALYST OVERRIDE ---
        if decision.anomaly_type == "volumetric":
            import re

            count_match = re.search(
                r"(\d+)\s+(?:object|item|cup|ball|bottle)", hypothesis
            )
            if not count_match:
                count_match = re.search(
                    r"(?:are|see|count|found|total)\D*(\d+)", hypothesis
                )
            if count_match:
                try:
                    slm_count = int(count_match.group(1))
                    if slm_count > 0 and slm_count != perception.n_visible:
                        logger.warning(
                            "[director] ANALYST OVERRIDE: Scout=%d vs SLM=%d. Trusting SLM.",
                            perception.n_visible,
                            slm_count,
                        )
                        updates["n_visible"] = slm_count
                        total_vol = perception.total_observed_volume
                        if total_vol > 0:
                            updates["estimated_unit_volume"] = total_vol / slm_count
                except Exception as e:
                    logger.warning("[director] SLM Override failed: %s", e)

    # --- INITIAL VOLUME ESTIMATION FROM SLM ---
    current_vol = perception.estimated_unit_volume
    if current_vol <= 0.001:
        target_list = updates.get(
            "active_intent", perception.active_intent or list(ctx.main_intent)
        )
        target_label = target_list[0] if target_list else "object"
        if slm:
            try:
                slm_vol = slm.estimate_object_volume(target_label)
                if slm_vol > 0 and slm_vol != 0.001:
                    updates["estimated_unit_volume"] = slm_vol
                    logger.info(
                        "[PHYSICAL PRIOR] SLM volume for '%s' is %.6f m^3",
                        target_label,
                        slm_vol,
                    )
            except Exception as e:
                logger.warning("[director] Failed to get volume prior: %s", e)

    # V3.3.4: Reset Freshness Circuit Breaker
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
        # 1. Extract semantic texture features (768-dim)
        features_tensor = dinov2.extract_features(perception.image)
        features = features_tensor.reshape(1, -1).numpy()

        # 2. Predict rho using MLP
        # The MLP expects (N, 768) features as trained in calibrate_heuristic()
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

    # --- FIX 4: Spatial Memory — Track which genesis intents are "accounted for" ---
    # Collect PRO genesis labels (non-contra) to track coverage
    genesis_pro_labels = [
        gi.get("label", "").lower()
        for gi in (perception.genesis_intents or [])
        if not gi.get("is_contra", False)
    ]
    accounted_labels = set()  # Labels that have at least one visual detection

    # FIX 12: Log loop phase for debugging
    loop_count = state["decision"].loop_count
    logger.info(
        "[v3_math_node] Phase: %s (loop_count=%d)",
        "DISCOVERY" if loop_count == 0 else "REFINEMENT",
        loop_count,
    )

    for cluster in filtered_clusters:
        c_vol = cluster.get("volume_m3", 0.0)
        # --- V3.8 Neuro-Symbolic Reconciliation (LNN) ---
        lnn_kb = get_lnn_kb()
        candidates = []
        for det in raw_detections:
            iou = MathUtils.calculate_bbox_overlap(
                cluster.get("bbox", {}), det.get("bbox", {})
            )
            if iou > 0.05:
                # Store source as grounding_dino for later arbitration
                candidates.append(
                    {
                        "label": det.get("label", "item"),
                        "iou": iou,
                        "source": "grounding_dino",
                    }
                )

        vjepa = get_vjepa_engine()
        # V4: Use Latent Anchors from Genesis Intents
        active_anchors = []
        for gi in perception.genesis_intents or []:
            l_pose = gi.get("latent_pose")
            if l_pose is not None and not gi.get("is_contra", False):
                active_anchors.append((gi.get("label", "item"), l_pose))

        # 0. Extract current cluster fingerprint (V-JEPA/DINOv2)
        cluster_latent = None
        if vjepa and cluster.get("bbox"):
            try:
                cluster_latent = vjepa.extract_object_latent(cluster.get("bbox"))
                if hasattr(cluster_latent, "detach"):
                    cluster_latent = cluster_latent.detach().cpu().numpy()
            except Exception as exc:
                logger.warning("[v3_math_node] Fingerprint extraction failed: %s", exc)

        # 1. Hybrid Discovery: Supplement GroundingDINO with Latent Search
        if cluster_latent is not None:
            for anchor_label, anchor_vec in active_anchors:
                # V4 Cosine Similarity (Strict)
                sim = np.dot(cluster_latent.flatten(), anchor_vec.flatten()) / (
                    np.linalg.norm(cluster_latent) * np.linalg.norm(anchor_vec) + 1e-9
                )
                if sim > 0.85:  # V4 High confidence threshold
                    exists = any(
                        c["label"].lower() == anchor_label.lower() for c in candidates
                    )
                    if not exists:
                        candidates.append(
                            {
                                "label": anchor_label,
                                "iou": float(sim),
                                "source": "latent_search",
                            }
                        )
                        logger.info(
                            "[v3_math_node] Identity Discovery: Cluster strictly matches anchor '%s' (sim=%.2f)",
                            anchor_label,
                            sim,
                        )

        best_identity_score = -1.0
        best_label = "item"
        best_iou = 0.0
        target_u_v = target_unit_v

        # V4.0 Specificity Guard: Replaces generic VLM labels with specific latent anchor labels
        GENERIC_LABELS = {
            "item",
            "thing",
            "product",
            "object",
            "can",
            "bottle",
            "drink",
            "items",
            "objects",
        }

        # Arbitrate between candidates using LNN Identity Rules
        for cand in candidates:
            cand_label = cand["label"]
            cand_iou = cand["iou"]

            # Fetch unit volume for this candidate
            u_v = slm.estimate_object_volume(cand_label) if slm else target_unit_v

            # 1. Volume Consistency Logic
            # FIX 10: Discovery-mode bypass for genesis pro-intents
            is_genesis_label = cand_label.lower() in genesis_pro_labels
            is_discovery_phase = (loop_count == 0)

            if is_discovery_phase and is_genesis_label:
                # DISCOVERY MODE: Skip volumetric validation
                # Rationale: Discovery only asks "does this exist?", not "how many?"
                vol_ratio = 1.0
                logger.info(
                    "[v3_math_node] DISCOVERY PASS: '%s' — volumetric check deferred to refinement",
                    cand_label,
                )
            else:
                # REFINEMENT MODE: Full volumetric validation (existing logic)
                n_est = c_vol / u_v if u_v > 0 else 1e9
                vol_ratio = 1.0
                if n_est > 2000:
                    vol_ratio = 0.1
                elif n_est < 0.3:
                    vol_ratio = 0.2

            # 2. Latent Match (DINOv2 Sovereignty)
            latent_match_score = cand_iou
            # If this wasn't discovered via latent search, verify its label using latents anyway
            if cluster_latent is not None and cand.get("source") != "latent_search":
                best_sim = 0.0
                best_match_label = None

                for anchor_label, anchor_vec in active_anchors:
                    if (
                        anchor_label.lower() in cand_label.lower()
                        or cand_label.lower() in anchor_label.lower()
                    ):
                        sim = np.dot(cluster_latent.flatten(), anchor_vec.flatten()) / (
                            np.linalg.norm(cluster_latent) * np.linalg.norm(anchor_vec)
                            + 1e-9
                        )
                        if sim > best_sim:
                            best_sim = float(sim)
                            best_match_label = anchor_label

                # V4.0: SPECIFICITY REPLACEMENT
                # If current label is generic but we have a specific latent match, override it
                if cand_label.lower() in GENERIC_LABELS and best_sim > 0.85:
                    logger.info(
                        "[v3_math_node] Specificity Guard: Overriding generic '%s' with specific anchor '%s' (sim=%.2f)",
                        cand_label,
                        best_match_label,
                        best_sim,
                    )
                    cand_label = best_match_label
                    # Re-fetch volume for the new specific label
                    u_v = (
                        slm.estimate_object_volume(cand_label) if slm else target_unit_v
                    )

                if best_sim > 0:
                    latent_match_score = best_sim
                    logger.debug(
                        "[v3_math_node] Latent re-verification for '%s': %.2f",
                        cand_label,
                        best_sim,
                    )

                # V4 Strict Semantic Collapse Guard:
                # If VLM says it's this object, but Latent similarity is bad (<0.85), veto it.
                # EXCEPTION: Bypass veto for known genesis pro-intent labels
                if best_sim < 0.85 and cand.get("source") == "grounding_dino":
                    is_genesis_label = cand_label.lower() in genesis_pro_labels
                    if not is_genesis_label:
                        logger.warning(
                            "[v3_math_node] Vetoing VLM label '%s' due to low latent sim %.2f",
                            cand_label,
                            best_sim,
                        )
                        latent_match_score = 0.0
                    else:
                        logger.info(
                            "[v3_math_node] Genesis Override: '%s' bypasses veto (sim=%.2f, in genesis=%s)",
                            cand_label,
                            best_sim,
                            is_genesis_label,
                        )
                        latent_match_score = 0.5  # Neutral: trust genesis but don't blindly accept

            # Ask LNN Model for the truth value of: Identity(cluster, sku)
            identity_score = lnn_kb.reconcile_identity(
                str(id(cluster)), cand_label, latent_match_score, vol_ratio
            )

            if identity_score > best_identity_score:
                best_identity_score = identity_score
                best_label = cand_label
                best_iou = cand_iou
                target_u_v = u_v

        # --- FIX 4 (cont.): Residual Intent Mapping ---
        # If this cluster has NO visual detection match or LNN rejected identity
        if best_identity_score < 0.3 and best_label == "item":
            # Find unaccounted genesis labels
            unaccounted = [
                gl for gl in genesis_pro_labels if gl not in accounted_labels
            ]
            if unaccounted:
                best_label = unaccounted[0]
                target_u_v = (
                    slm.estimate_object_volume(best_label) if slm else target_unit_v
                )
                logger.info(
                    "[v3_math_node] SPATIAL MEMORY: Unmatched cluster → "
                    "assigned to unaccounted genesis intent '%s' (occluded?)",
                    best_label,
                )
        else:
            # Track that this label is visually accounted for
            for gl in genesis_pro_labels:
                if gl in best_label.lower() or best_label.lower() in gl:
                    accounted_labels.add(gl)

        # Calculate fractional count with LNN-validated unit volume
        u_v = target_u_v
        cluster_n = (c_vol * perception.rho) / u_v if u_v > 1e-9 else 0.0
        n_vol += cluster_n

        logger.info(
            "[v3_math_node] LNN Identity: %s (score=%.2f) -> V=%.6f m^3 -> n=%.2f (iou=%.2f)",
            best_label,
            best_identity_score,
            c_vol,
            cluster_n,
            best_iou,
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
        # V5.1: Instead of forcing exit, allow perception to continue (Wait-and-Watch)
        if (
            gate_decision.anomaly_type.value == "volumetric"
            and not perception.is_volumetric_data_fresh
        ):
            logger.warning(
                "[logic_gate_node] CIRCUIT BREAKER: Volumetric anomaly with STALE data detected. Continuing perception (Wait-and-Watch)."
            )
            return {
                "decision": {
                    "status": "continue_perception",
                    "anomaly_type": "none",
                    "logic_gate_result": {
                        "rule_applied": "CIRCUIT_BREAKER_STALE_DATA",
                        "confidence": 0.0,
                        "action": "continue_perception",
                        "reasoning": "Circuit Breaker: Volumetric data is stale (V3 Math did not update). Bypassing SLM and continuing perception accumulation.",
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
) -> Literal["exit", "targeted_slm_node", "interpolation_node"]:
    """
    Conditional edge: Route based on Logic Gate decision.
    
    V5.1: Added 'continue_perception' route to bypass SLM when volumetric data is stale.
    """
    decision = state["decision"]

    logger.info("[route] Checking decision status: %s", decision.status)

    if decision.status == "exit":
        return "exit"
    elif decision.loop_count >= state["ctx"].max_loop_count:
        logger.warning("[route] Max loop count reached, forcing exit")
        return "exit"
    elif decision.status == "continue_perception":
        # V5.1: Bypass SLM and route directly to interpolation → latent_director
        logger.info("[route] Continue perception detected, bypassing SLM")
        return "interpolation_node"
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
    latent_director_node (Step 2: Director)
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
                 ┌───��────────┴────────────┐
                 v                         v
               (exit)              targeted_slm_node (Step 12)
                 │                         │
                 v                         v
                END               interpolation_node
                                           │
                                           v
                                  latent_director_node (loop)
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
    graph.add_node("latent_director_node", latent_director_node)
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
    graph.add_edge("vjepa_brain_node", "latent_director_node")

    # Director → Parallel Senses (Sovereignties 1, 2, 3)
    graph.add_edge("latent_director_node", "v2e_sensor_node")
    graph.add_edge("latent_director_node", "sam2_depth_node")
    graph.add_edge("latent_director_node", "countvid_executor_node")
    graph.add_edge("latent_director_node", "density_sensing_node")

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
    # V5.1: Added 'interpolation_node' route for 'continue_perception' status
    graph.add_conditional_edges(
        "logic_gate_node",
        route_after_logic_gate,
        {
            "exit": END,
            "targeted_slm_node": "targeted_slm_node",
            "interpolation_node": "interpolation_node",
        },
    )

    # SLM → Interpolation → Director (Recursive Loop)
    graph.add_edge("targeted_slm_node", "interpolation_node")
    graph.add_edge("interpolation_node", "latent_director_node")

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
