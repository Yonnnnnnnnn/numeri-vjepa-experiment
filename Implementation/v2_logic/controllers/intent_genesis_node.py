"""
Intent Genesis Node (Step 0) — V3.5 Global Pre-Scan

Pre-emptive semantic scanning that runs ONCE before the main recursive loop.
Uses GroundingDINO (Scout) to locate objects across ALL pre-scanned video
keyframes, then VLM Qwen (Analyst) to produce SKU-specific intent labels
and reference crop images.

V3.5: Genesis now receives `perception.video_frames` — keyframes sampled from
the ENTIRE video duration (~1 frame/2sec). This ensures the system discovers
ALL brands/variants that appear anywhere in the video, not just the first frame.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : IntentGenesisModule (this file)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <intent_genesis_node>       → Main LangGraph node function               │
  │  <_sample_frames>            → Sparse video sampling helper               │
  │  <_run_grounding_scout_all>  → GroundingDINO on ALL sampled frames        │
  │  <_run_vlm_analyst>          → VLM classification (single frame)          │
  │  <_run_vlm_analyst_multi>    → Multi-frame VLM + dedup (V3.5)             │
  │  <_crop_detections>          → Extract reference crop images              │
  │  <_extract_latent_anchors>   → V3.7 Accumulative Fingerprinting           │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <RecursiveFlowState>  ← from types.graph_state (State Container)         │
  │  <SLMEngine>           ← from models.slm_engine (VLM Analyst)             │
  │  <CountVidEngine>      ← from models.count_vid_engine (GroundingDINO)     │
  │  <cv2>                 ← from opencv-python (Image processing)            │
  │  <torch>               ← from torch (Deep learning framework)             │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : str, dict, list, numpy.ndarray, int, float

Production Rules:
  IntentGenesisModule  → imports + <intent_genesis_node>
  intent_genesis_node  → _sample_frames + _run_grounding_scout_all
                       + _run_vlm_analyst_multi + _crop_detections
                       → genesis_intents + reference_crops
═══════════════════════════════════════════════════════════════════════════════

Pattern: Template Method
- Defines the skeleton of intent generation (sample → scout → analyze → crop).
- Subcomponents (GroundingDINO, VLM) are pluggable strategies.

V3.3: This node replaces hardcoded discovery_keywords with dynamic intent generation.
V3.5: Global Pre-Scan — multi-frame Genesis across entire video duration.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import cv2

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


def _run_grounding_scout_all(
    sampled_frames: List[Tuple[int, np.ndarray]],
    user_prompt: str,
    countvid_engine: Any,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    V3.5: Run GroundingDINO on ALL sampled frames and return every frame
    that has detections, sorted by confidence descending.

    This replaces the old _run_grounding_scout which only returned the
    single best-confidence frame, causing "First-Frame Bias".

    Args:
        sampled_frames: List of (frame_idx, frame) from sparse sampling.
        user_prompt: Raw user intent (e.g., "count soda").
        countvid_engine: CountVidEngine instance (wraps GroundingDINO).
        max_results: Maximum number of frames to return.

    Returns:
        List of dicts, each with 'frame_idx', 'frame', 'detections',
        'count', 'avg_confidence'. Sorted by confidence descending.
    """
    if not sampled_frames or not countvid_engine:
        return []

    all_results = []

    for frame_idx, frame in sampled_frames:
        try:
            count_val, raw_detections = countvid_engine.count(
                frame, [user_prompt], target_size=(frame.shape[1], frame.shape[0])
            )

            if count_val > 0 and raw_detections:
                avg_conf = sum(
                    det.get("score", 0.5) if isinstance(det, dict) else 0.5
                    for det in raw_detections
                ) / len(raw_detections)

                all_results.append(
                    {
                        "frame_idx": frame_idx,
                        "frame": frame,
                        "detections": raw_detections,
                        "count": count_val,
                        "avg_confidence": avg_conf,
                    }
                )

        except Exception as exc:
            logger.warning(
                "[IntentGenesis] Scout failed on frame %d: %s", frame_idx, exc
            )
            continue

    # Sort by confidence descending, take top-N
    all_results.sort(key=lambda r: r["avg_confidence"], reverse=True)
    return all_results[:max_results]


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
        logger.warning(
            "[IntentGenesis] SLM.generate_initial_intents not available, "
            "using raw prompt fallback"
        )
        return [{"label": user_prompt, "confidence": 0.5, "source": "fallback"}]
    except Exception as exc:
        logger.error("[IntentGenesis] VLM Analyst failed: %s", exc)
        return [{"label": user_prompt, "confidence": 0.3, "source": "error_fallback"}]


def _run_vlm_analyst_multi(
    scout_results: List[Dict[str, Any]],
    user_prompt: str,
    slm_engine: Any,
    max_analysis_frames: int = 5,
) -> List[Dict[str, Any]]:
    """
    V3.5: Run VLM analysis on multiple scout frames and merge/deduplicate
    discovered SKU labels into a single global manifest.

    This prevents "First-Frame Bias" by examining frames from throughout
    the entire video duration.

    Args:
        scout_results: List of scout result dicts from _run_grounding_scout_all.
        user_prompt: Raw user intent.
        slm_engine: SLMEngine instance wrapping VLM.
        max_analysis_frames: Max frames to analyze (controls VLM compute cost).

    Returns:
        Deduplicated list of genesis intent dicts.
    """
    if not scout_results:
        return [{"label": user_prompt, "confidence": 0.3, "source": "no_scout"}]

    all_intents = []
    seen_labels = set()

    for result in scout_results[:max_analysis_frames]:
        frame_intents = _run_vlm_analyst(
            frame=result["frame"],
            detections=result["detections"],
            user_prompt=user_prompt,
            slm_engine=slm_engine,
        )

        for intent in frame_intents:
            normalized = intent.get("label", "").strip().lower()
            if not normalized:
                continue
            if normalized not in seen_labels:
                seen_labels.add(normalized)
                all_intents.append(intent)
                logger.info(
                    "[IntentGenesis] NEW SKU discovered at frame %d: '%s'",
                    result["frame_idx"],
                    intent.get("label", "?"),
                )

    if not all_intents:
        return [{"label": user_prompt, "confidence": 0.3, "source": "vlm_empty"}]

    logger.info(
        "[IntentGenesis] Multi-frame VLM found %d unique SKUs: %s",
        len(all_intents),
        [i.get("label", "?") for i in all_intents],
    )
    return all_intents


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


def _extract_latent_anchors(
    scout_results: List[Dict[str, Any]],
    genesis_intents: List[Dict[str, Any]],
    user_prompt: str,
    vjepa_engine: Any,
) -> Dict[str, np.ndarray]:
    """
    V3.7: Extract Latent Anchors from MULTIPLE top frames (Accumulative Fingerprinting).

    Matches scouted detections against VLM-discovered SKU labels to create
    high-dimensional visual anchors for latent matching.

    Args:
        scout_results: Results from GroundingDINO scout.
        genesis_intents: Discovered SKU labels from VLM.
        user_prompt: Fallback label.
        vjepa_engine: VJEPAEngine instance.

    Returns:
        Mapping of SKU labels to latent vectors (numpy).
    """
    latent_anchors = {}
    if not vjepa_engine:
        return {}

    try:
        # Analyze top N frames for latent anchors (usually those with highest diversity)
        for result in scout_results[:3]:
            frame = result["frame"]
            frame_detections = result["detections"]

            # Prepare frame tensor [1, 3, 224, 224] for V-JEPA (V3.7 Standard)
            # Use specific dtypes to help static analysis
            frame_resized = cv2.resize(frame, (224, 224))
            frame_tensor = (
                torch.from_numpy(frame_resized).permute(2, 0, 1).float().unsqueeze(0)
                / 255.0
            ).to(vjepa_engine.device)

            vjepa_latent_map = vjepa_engine.encode(frame_tensor)

            for det_item in frame_detections:
                # Handle GroundingDINO list format [x1, y1, x2, y2]
                if isinstance(det_item, list) and len(det_item) >= 4:
                    det_bbox = {
                        "x1": det_item[0],
                        "y1": det_item[1],
                        "x2": det_item[2],
                        "y2": det_item[3],
                    }
                    h, w = frame.shape[:2]
                    norm_bbox = {
                        "x": det_bbox["x1"] / w,
                        "y": det_bbox["y1"] / h,
                        "w": (det_bbox["x2"] - det_bbox["x1"]) / w,
                        "h": (det_bbox["y2"] - det_bbox["y1"]) / h,
                    }
                    det = {"label": user_prompt, "bbox": norm_bbox}
                else:
                    det = det_item

                label = det.get("label", "object").lower()
                det_words = set(label.replace(",", " ").split())

                # Match detection label with specific Genesis labels
                matched_sku = None
                for intent in genesis_intents:
                    sku_label = intent["label"].lower()
                    sku_words = set(sku_label.replace(",", " ").split())

                    if (
                        det_words & sku_words
                        or label in sku_label
                        or sku_label in label
                    ):
                        matched_sku = intent["label"]
                        if len(sku_words) > 1:
                            break  # Specific match found

                if matched_sku and matched_sku not in latent_anchors:
                    latent_vec = vjepa_engine.extract_object_latent(
                        det["bbox"], vjepa_latent_map
                    )
                    latent_anchors[matched_sku] = latent_vec.detach().cpu().numpy()
                    logger.info(
                        "[IntentGenesis] Stored V3.7 Latent Anchor for SKU: '%s'",
                        matched_sku,
                    )
    except Exception as exc:
        logger.error("[IntentGenesis] Failed to extract V3.7 Latent Anchors: %s", exc)

    return latent_anchors


# =============================================================================
# MAIN NODE FUNCTION
# =============================================================================


def intent_genesis_node(state: RecursiveFlowState) -> Dict[str, Any]:
    """
    Step 0: Intent Genesis — V3.5 Global Pre-Scan.

    Runs ONCE at the start of the recursive flow to produce:
    1. genesis_intents: SKU-specific intent list from ENTIRE video.
    2. reference_crops: Visual anchors for V-JEPA latent matching.
    3. active_intent: Populated immediately for parallel processing.

    V3.5: Uses `perception.video_frames` (pre-scanned keyframes from the
    full video) to discover ALL brands/variants, not just what appears
    in the first frame.
    """
    logger.info("[intent_genesis_node] === STEP 0: Intent Genesis (V3.5 Global) ===")

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

    # --- Phase 1: Determine frame source ---
    # V3.5: Prefer video_frames (full pre-scan) over single image
    frames = []
    if perception.video_frames:
        frames = perception.video_frames
        logger.info(
            "[intent_genesis_node] V3.5 GLOBAL mode: %d pre-scanned keyframes",
            len(frames),
        )
    elif perception.image is not None:
        frames = [(0, perception.image)]
        logger.info("[intent_genesis_node] Single-frame fallback mode")
    else:
        logger.warning("[intent_genesis_node] No frames available for genesis")
        fallback_intents = [
            {"label": user_prompt, "confidence": 0.5, "source": "no_frame"}
        ]
        return {
            "perception": {
                "genesis_intents": fallback_intents,
                "active_intent": [user_prompt],
            },
            "decision": {"is_genesis_complete": True},
        }

    # --- Phase 2: Scout with GroundingDINO across ALL frames ---
    from .recursive_flow import get_countvid_engine, get_slm_engine

    countvid = get_countvid_engine()
    scout_results = _run_grounding_scout_all(frames, user_prompt, countvid)

    if not scout_results:
        logger.warning(
            "[intent_genesis_node] Scout found nothing in any frame, "
            "using prompt as intent"
        )
        fallback_intents = [
            {"label": user_prompt, "confidence": 0.3, "source": "scout_empty"}
        ]
        return {
            "perception": {
                "genesis_intents": fallback_intents,
                "active_intent": [user_prompt],
            },
            "decision": {"is_genesis_complete": True},
        }

    total_detections = sum(r["count"] for r in scout_results)
    logger.info(
        "[intent_genesis_node] Scout found objects in %d/%d frames (%d total detections)",
        len(scout_results),
        len(frames),
        total_detections,
    )

    # --- Phase 3: Multi-frame VLM Analysis (V3.5) ---
    slm = get_slm_engine()
    genesis_intents = _run_vlm_analyst_multi(
        scout_results, user_prompt, slm, max_analysis_frames=5
    )

    # --- Phase 4: Extract Reference Crops & Latent Anchors ---
    from .recursive_flow import get_vjepa_engine
    import torch

    best_frame = scout_results[0]["frame"]
    best_detections = scout_results[0]["detections"]
    reference_crops = _crop_detections(best_frame, best_detections)

    # V3.7: Extract Latent Anchors from MULTIPLE top frames (Accumulative Fingerprinting)
    from .recursive_flow import get_vjepa_engine

    vjepa = get_vjepa_engine()
    latent_anchors = _extract_latent_anchors(
        scout_results, genesis_intents, user_prompt, vjepa
    )

    # --- Phase 5: Populate active_intent from genesis results ---
    active_labels = []
    for intent in genesis_intents:
        label = intent.get("label", "")
        if label and label not in active_labels:
            active_labels.append(label)

    # Ensure main_intent items are included (but avoid generic duplicates)
    for item in ctx.main_intent:
        if item not in active_labels:
            active_labels.append(item)

    logger.info(
        "[intent_genesis_node] Genesis complete: %d intents, %d crops, %d latent anchors → %s",
        len(genesis_intents),
        len(reference_crops),
        len(latent_anchors),
        active_labels,
    )

    return {
        "perception": {
            "genesis_intents": genesis_intents,
            "reference_crops": reference_crops,
            "active_intent": active_labels,
            "latent_anchors": latent_anchors,
        },
        "decision": {"is_genesis_complete": True},
    }
