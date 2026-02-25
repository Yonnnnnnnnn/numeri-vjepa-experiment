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
# VLM GROUNDED SCOUT (V3.9)
# =============================================================================


def _run_vlm_scout_and_analyst(
    sampled_frames: List[Tuple[int, np.ndarray]],
    user_prompt: str,
    slm_engine: Any,
    max_frames: int = 3,
) -> List[Dict[str, Any]]:
    """
    V3.9: Use VLM directly to scout and analyze frames simultaneously.
    This eliminates GroundingDINO's prompt overload issue.

    Args:
        sampled_frames: Sampled video keyframes.
        user_prompt: Raw user intent.
        slm_engine: SLMEngine instance.
        max_frames: Max frames to analyze.

    Returns:
        Deduplicated list of genesis intent dicts with 'bbox' and 'frame'.
    """
    if not sampled_frames or not slm_engine:
        return []

    all_intents = []
    seen_labels = set()

    # Analyze top frames (usually start, middle, end or based on diversity)
    for idx, frame in sampled_frames[:max_frames]:
        try:
            logger.info("[IntentGenesis] VLM analyzing frame %d for SKUs...", idx)
            frame_intents = slm_engine.generate_initial_intents(
                prompt=user_prompt,
                frame=frame,
            )

            for intent in frame_intents:
                label = intent.get("label", "").lower().strip()
                if not label:
                    continue

                # Add frame reference for latent extraction
                intent["frame"] = frame
                intent["frame_idx"] = idx

                if label not in seen_labels:
                    seen_labels.add(label)
                    all_intents.append(intent)
                    logger.info(
                        "[IntentGenesis] VLM GROUNDED discovery: '%s' at frame %d",
                        label,
                        idx,
                    )

        except Exception as exc:
            logger.error("[IntentGenesis] VLM scout failed on frame %d: %s", idx, exc)

    return all_intents


# VLM Analyst helpers removed in V3.9 as they are integrated into _run_vlm_scout_and_analyst


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
    genesis_intents: List[Dict[str, Any]],
    vjepa_engine: Any,
) -> Dict[str, np.ndarray]:
    """
    V3.9: Extract Latent Anchors using direct VLM-grounded bounding boxes.
    Uses DINOv2 fingerprints for multi-shield consistency.

    Args:
        genesis_intents: List of discovered intents with 'bbox' and 'frame'.
        vjepa_engine: VJEPAEngine instance to provide latent context.

    Returns:
        Mapping: {label: latent_vector}
    """
    latent_anchors = {}
    if not vjepa_engine:
        return {}

    try:
        # Sort intents by frame to prevent redundant encoding
        from collections import defaultdict

        frame_groups = defaultdict(list)
        for intent in genesis_intents:
            if "frame" in intent and intent.get("bbox"):
                frame_groups[id(intent["frame"])].append(intent)

        for frame_id, intents in frame_groups.items():
            frame = intents[0]["frame"]

            # Standard 224x224 resize for V-JEPA
            frame_resized = cv2.resize(frame, (224, 224))
            frame_tensor = (
                torch.from_numpy(frame_resized).permute(2, 0, 1).float().unsqueeze(0)
                / 255.0
            ).to(vjepa_engine.device)

            # Extract full-frame latent map
            vjepa_latent_map = vjepa_engine.encode(frame_tensor)

            for intent in intents:
                label = intent["label"]
                bbox = intent["bbox"]

                if label not in latent_anchors:
                    # Extract 768-D fingerprint from V-JEPA context
                    latent_vec = vjepa_engine.extract_object_latent(
                        bbox, vjepa_latent_map
                    )
                    latent_anchors[label] = latent_vec.detach().cpu().numpy()
                    logger.info("[IntentGenesis] Fingerprinted SKU: '%s'", label)

    except Exception as exc:
        logger.error("[IntentGenesis] Fingerprinting failed: %s", exc)

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
    if perception.video_frames:
        frames = perception.video_frames
    elif perception.image is not None:
        frames = [(0, perception.image)]
    else:
        logger.warning("[intent_genesis_node] No frames available for genesis")
        return {
            "perception": {
                "genesis_intents": [],
                "active_intent": list(ctx.main_intent),
            },
            "decision": {"is_genesis_complete": True},
        }

    # --- Phase 2: VLM Grounded Discovery (DINOv2 x VLM Overhaul V3.9) ---
    from .recursive_flow import get_slm_engine, get_vjepa_engine

    slm = get_slm_engine()
    vjepa = get_vjepa_engine()

    # Step A: Perform Grounded Intent Discovery
    genesis_intents = _run_vlm_scout_and_analyst(frames, user_prompt, slm)

    # Step B: Perform Accumulative Visual Fingerprinting
    latent_anchors = _extract_latent_anchors(genesis_intents, vjepa)

    # Step C: Extract Reference Crops for Visualizer
    reference_crops = []
    if genesis_intents:
        # Use bboxes from genesis intents to create crops from their parent frames
        for intent in genesis_intents[:5]:
            if intent.get("bbox") and "frame" in intent:
                f = intent["frame"]
                h, w = f.shape[:2]
                b = intent["bbox"]
                x1, y1 = int(b["x"] * w), int(b["y"] * h)
                x2, y2 = int((b["x"] + b["w"]) * w), int((b["y"] + b["h"]) * h)
                crop = f[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)].copy()
                reference_crops.append(crop)

    # --- Phase 3: Populate active_intent ---
    active_labels = [i["label"] for i in genesis_intents if not i.get("is_contra")]

    # Add main_intent from context if not already present
    for m in ctx.main_intent:
        if m not in active_labels:
            active_labels.append(m)

    logger.info(
        "[intent_genesis_node] V3.9 Genesis Complete: %d SKUs → %s",
        len(active_labels),
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
