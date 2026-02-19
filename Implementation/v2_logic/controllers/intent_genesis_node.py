"""
Intent Genesis Node (Step 0)

Pre-emptive semantic scanning that runs ONCE before the main recursive loop.
Uses GroundingDINO (Scout) to locate objects and VLM Qwen (Analyst)
to produce SKU-specific intent labels and reference crop images.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : IntentGenesisModule (this file)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <intent_genesis_node>  → Main LangGraph node function                    │
  │  <_sample_frames>       → Sparse video sampling helper                    │
  │  <_run_grounding_scout> → GroundingDINO detection on sampled frames       │
  │  <_run_vlm_analyst>     → VLM classification of detected objects          │
  │  <_crop_detections>     → Extract reference crop images from detections   │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <RecursiveFlowState>  ← from types.graph_state (State Container)         │
  │  <SLMEngine>           ← from models.slm_engine (VLM Analyst)             │
  │  <CountVidEngine>      ← from models.count_vid_engine (GroundingDINO)     │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : str, dict, list, numpy.ndarray, int, float

Production Rules:
  IntentGenesisModule  → imports + <intent_genesis_node>
  intent_genesis_node  → _sample_frames + _run_grounding_scout
                       + _run_vlm_analyst + _crop_detections
                       → genesis_intents + reference_crops
═══════════════════════════════════════════════════════════════════════════════

Pattern: Template Method
- Defines the skeleton of intent generation (sample → scout → analyze → crop).
- Subcomponents (GroundingDINO, VLM) are pluggable strategies.

V3.3: This node replaces hardcoded discovery_keywords with dynamic intent generation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..types.graph_state import RecursiveFlowState

logger = logging.getLogger(__name__)

# =============================================================================
# SPARSE SAMPLING HELPER
# =============================================================================


def _sample_frames(
    video_frames: List[np.ndarray],
    sample_rate: int = 30,
    max_samples: int = 10,
) -> List[Tuple[int, np.ndarray]]:
    """
    Sparse-sample video frames for efficiency.

    Args:
        video_frames: Full list of video frames.
        sample_rate: Take 1 frame every N frames (~1fps at 30fps video).
        max_samples: Maximum number of frames to sample.

    Returns:
        List of (frame_index, frame) tuples.
    """
    if not video_frames:
        return []

    indices = list(range(0, len(video_frames), sample_rate))[:max_samples]
    return [(idx, video_frames[idx]) for idx in indices]


# =============================================================================
# GROUNDING SCOUT (GroundingDINO)
# =============================================================================


def _run_grounding_scout(
    sampled_frames: List[Tuple[int, np.ndarray]],
    user_prompt: str,
    countvid_engine: Any,
) -> Optional[Dict[str, Any]]:
    """
    Run GroundingDINO on sampled frames to find the highest-confidence detection.

    Args:
        sampled_frames: List of (frame_idx, frame) from sparse sampling.
        user_prompt: Raw user intent (e.g., "count soda").
        countvid_engine: CountVidEngine instance (wraps GroundingDINO).

    Returns:
        Dict with 'frame_idx', 'frame', 'detections', 'best_confidence',
        or None if no detections found.
    """
    if not sampled_frames or not countvid_engine:
        return None

    best_result = None
    best_confidence = 0.0

    for frame_idx, frame in sampled_frames:
        try:
            count_val, raw_detections = countvid_engine.count(
                frame, [user_prompt], target_size=(frame.shape[1], frame.shape[0])
            )

            if count_val > 0 and raw_detections:
                # Calculate average confidence for this frame
                avg_conf = sum(
                    det.get("score", 0.5) if isinstance(det, dict) else 0.5
                    for det in raw_detections
                ) / len(raw_detections)

                if avg_conf > best_confidence:
                    best_confidence = avg_conf
                    best_result = {
                        "frame_idx": frame_idx,
                        "frame": frame,
                        "detections": raw_detections,
                        "count": count_val,
                        "best_confidence": avg_conf,
                    }

        except Exception as exc:
            logger.warning(
                "[IntentGenesis] Scout failed on frame %d: %s", frame_idx, exc
            )
            continue

    return best_result


# =============================================================================
# VLM ANALYST (Qwen)
# =============================================================================


def _run_vlm_analyst(
    frame: np.ndarray,
    detections: List[Any],
    user_prompt: str,
    slm_engine: Any,
) -> List[Dict[str, Any]]:
    """
    Use VLM to analyze detected objects and produce specific SKU names.

    Args:
        frame: The best frame from GroundingDINO scouting.
        detections: Raw detections from GroundingDINO.
        user_prompt: Raw user intent.
        slm_engine: SLMEngine instance wrapping VLM.

    Returns:
        List of genesis intent dicts: [{'label': str, 'confidence': float, ...}].
    """
    if slm_engine is None:
        # Fallback: use user_prompt directly as intent
        logger.warning("[IntentGenesis] No SLM available, using raw prompt as intent")
        return [{"label": user_prompt, "confidence": 0.5, "source": "fallback"}]

    try:
        result = slm_engine.generate_initial_intents(
            prompt=user_prompt,
            frame=frame,
            detections=detections,
        )
        return result
    except AttributeError:
        # generate_initial_intents not yet implemented — graceful fallback
        logger.warning(
            "[IntentGenesis] SLM.generate_initial_intents not available, "
            "using raw prompt fallback"
        )
        return [{"label": user_prompt, "confidence": 0.5, "source": "fallback"}]
    except Exception as exc:
        logger.error("[IntentGenesis] VLM Analyst failed: %s", exc)
        return [{"label": user_prompt, "confidence": 0.3, "source": "error_fallback"}]


# =============================================================================
# CROP EXTRACTION
# =============================================================================


def _crop_detections(
    frame: np.ndarray,
    detections: List[Any],
    padding: int = 10,
) -> List[np.ndarray]:
    """
    Extract cropped reference images from detections.

    Args:
        frame: Source frame (H, W, 3).
        detections: List of detection dicts with bbox info.
        padding: Pixel padding around crops.

    Returns:
        List of cropped numpy arrays.
    """
    crops = []
    h, w = frame.shape[:2]

    for det in detections:
        try:
            # Handle both list format [x1,y1,x2,y2] and dict format
            if isinstance(det, list) and len(det) >= 4:
                x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            elif isinstance(det, dict):
                bbox = det.get("bbox", det)
                if isinstance(bbox, dict):
                    x1 = int(bbox.get("x", 0))
                    y1 = int(bbox.get("y", 0))
                    x2 = int(x1 + bbox.get("w", 0))
                    y2 = int(y1 + bbox.get("h", 0))
                elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    x1, y1, x2, y2 = (
                        int(bbox[0]),
                        int(bbox[1]),
                        int(bbox[2]),
                        int(bbox[3]),
                    )
                else:
                    continue
            else:
                continue

            # Apply padding with bounds clamping
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2].copy()
                crops.append(crop)

        except Exception as exc:
            logger.warning("[IntentGenesis] Crop extraction failed: %s", exc)
            continue

    return crops


# =============================================================================
# MAIN NODE FUNCTION
# =============================================================================


def intent_genesis_node(state: RecursiveFlowState) -> Dict[str, Any]:
    """
    Step 0: Intent Genesis — Pre-emptive Semantic Scanning.

    Runs ONCE at the start of the recursive flow to produce:
    1. genesis_intents: SKU-specific intent list (replaces hardcoded keywords).
    2. reference_crops: Visual anchors for V-JEPA latent matching.
    3. active_intent: Populated immediately for parallel processing.

    If video frames are not available (single-frame mode), falls back to
    using the user prompt directly as the intent.
    """
    logger.info("[intent_genesis_node] === STEP 0: Intent Genesis ===")

    ctx = state["ctx"]
    perception = state["perception"]
    user_prompt = ctx.user_prompt

    # Guard: If no user prompt, use main_intent as fallback
    if not user_prompt and ctx.main_intent:
        user_prompt = " ".join(ctx.main_intent)
        logger.info(
            "[intent_genesis_node] No user_prompt; derived from main_intent: '%s'",
            user_prompt,
        )

    if not user_prompt:
        logger.warning(
            "[intent_genesis_node] No user prompt available, skipping genesis"
        )
        return {}

    # --- Phase 1: Check if we have video frames or just a single image ---
    # In single-frame mode, we use the current image directly
    frames = []
    if perception.image is not None:
        frames = [(0, perception.image)]
        logger.info("[intent_genesis_node] Single-frame mode: using current image")
    else:
        logger.warning("[intent_genesis_node] No frames available for genesis")
        # Fallback: populate active_intent from user prompt
        fallback_intents = [
            {"label": user_prompt, "confidence": 0.5, "source": "no_frame"}
        ]
        updated = perception.model_copy(
            update={
                "genesis_intents": fallback_intents,
                "active_intent": [user_prompt],
            }
        )
        return {"perception": updated}

    # --- Phase 2: Scout with GroundingDINO ---
    from .recursive_flow import get_countvid_engine, get_slm_engine

    countvid = get_countvid_engine()
    scout_result = _run_grounding_scout(frames, user_prompt, countvid)

    if scout_result is None:
        logger.warning(
            "[intent_genesis_node] Scout found nothing, using prompt as intent"
        )
        fallback_intents = [
            {"label": user_prompt, "confidence": 0.3, "source": "scout_empty"}
        ]
        updated = perception.model_copy(
            update={
                "genesis_intents": fallback_intents,
                "active_intent": [user_prompt],
            }
        )
        return {"perception": updated}

    best_frame = scout_result["frame"]
    raw_detections = scout_result["detections"]

    logger.info(
        "[intent_genesis_node] Scout found %d objects at frame %d (conf=%.2f)",
        scout_result["count"],
        scout_result["frame_idx"],
        scout_result["best_confidence"],
    )

    # --- Phase 3: Analyze with VLM ---
    slm = get_slm_engine()
    genesis_intents = _run_vlm_analyst(best_frame, raw_detections, user_prompt, slm)

    # --- Phase 4: Extract Reference Crops ---
    reference_crops = _crop_detections(best_frame, raw_detections)

    # --- Phase 5: Populate active_intent from genesis results ---
    active_labels = []
    for intent in genesis_intents:
        label = intent.get("label", "")
        if label and label not in active_labels:
            active_labels.append(label)

    # Ensure main_intent items are included as well
    for item in ctx.main_intent:
        if item not in active_labels:
            active_labels.append(item)

    logger.info(
        "[intent_genesis_node] Genesis complete: %d intents, %d crops → %s",
        len(genesis_intents),
        len(reference_crops),
        active_labels,
    )

    return {
        "perception": {
            "genesis_intents": genesis_intents,
            "reference_crops": reference_crops,
            "active_intent": active_labels,
        },
        "decision": {"is_genesis_complete": True},
    }
