"""
Run Recursive Intent System

Main execution script for the end-to-end Recursive Intent system.
Orchestrates LangGraph execution on real video data.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : RunRecursiveSystem (this script)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <MainLoop>        → Frame-by-frame orchestration loop                    │
  │  <InitialState>    → State initialization                                 │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <RecursiveFlow>   ← from v2_logic.controllers.recursive_flow             │
  │  <GraphState>       ← from v2_logic.types.graph_state                      │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : cv2, numpy, time, logging

Production Rules:
  RunRecursiveSystem → setup_logging + <InitialState> + <MainLoop>
═══════════════════════════════════════════════════════════════════════════════

Pattern: Command (Script)
- Acts as the main entry point to invoke the recursive intent pipeline.
"""

import os
import sys

# 🩹 Emergency Patch for Torchvision Circular Import (StochasticDepth)
# This MUST happen before any torchvision/torch imports
try:
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.ops

    if not hasattr(torchvision.ops, "StochasticDepth"):

        class StochasticDepth(nn.Module):
            def __init__(self, p=0.1, mode="batch"):
                super().__init__()

            def forward(self, x):
                return x

        torchvision.ops.StochasticDepth = StochasticDepth
except Exception:
    pass

import logging
import cv2
import time
import numpy as np
import argparse

# Add Implementation root to path
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Robustly add Techs dependencies (required for Colab subprocesses)
TECHS_PATHS = [
    os.path.join(PARENT_DIR, "Techs/Depth-Anything-V2-main/Depth-Anything-V2-main"),
    os.path.join(PARENT_DIR, "Techs/CountVid-main/CountVid-main"),
    os.path.join(PARENT_DIR, "Techs/sam2-main/sam2-main"),
    os.path.join(PARENT_DIR, "Techs/v2e-master/v2e-master"),
]

for p in TECHS_PATHS:
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)

from v2_logic.controllers.recursive_flow import create_recursive_flow_app
from v2_logic.types.graph_state import create_initial_state


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    setup_logging()
    logger = logging.getLogger("run_recursive_system")

    # 0. Parse Arguments
    parser = argparse.ArgumentParser(description="Run Recursive Intent System")
    parser.add_argument(
        "--video",
        type=str,
        help="Path to input video file",
        default="Techs/sam2-main/sam2-main/demo/data/gallery/02_cups.mp4",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="User prompt for Intent Genesis Step 0 (e.g. 'count red cups')",
        default="",
    )
    args = parser.parse_args()

    # 1. Initialize Graph
    logger.info("Building Recursive Intent Graph...")
    app = create_recursive_flow_app()

    # 2. Input Source
    video_path = os.path.abspath(args.video)
    if not os.path.exists(video_path):
        # Specific fallback for Colab if default path fails
        colab_fallback = "/content/numeri-vjepa-experiment/Techs/sam2-main/sam2-main/demo/data/gallery/02_cups.mp4"
        if os.path.exists(colab_fallback):
            video_path = colab_fallback
        else:
            logger.error(f"Video not found: {video_path}")
            return

    cap = cv2.VideoCapture(video_path)  # pylint: disable=no-member
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # pylint: disable=no-member

    # 3. Initial Configuration
    session_id = f"session_{int(time.time())}"
    user_prompt = args.prompt.strip() if args.prompt else ""

    # V3.9: Derive initial intent from prompt; Step 0 will refine it
    if user_prompt:
        # Extract the last word as a seed intent (e.g. "count red cups" → "cups")
        seed_words = user_prompt.lower().split()
        # Filter out common verbs
        skip_verbs = {"count", "find", "detect", "track", "show", "identify"}
        target_intent = [w for w in seed_words if w not in skip_verbs] or ["object"]
    else:
        target_intent = ["object"]

    initial_state = create_initial_state(
        session_id,
        target_intent,
        time.time(),
        user_prompt=user_prompt,
    )

    logger.info("Running Recursive Intent System - Logic Version: V5.1")
    logger.info(f"Session started: {session_id}")
    logger.info(f"User Prompt: '{user_prompt}'")
    logger.info(f"Seed Intent: {target_intent}")

    # ══════════════════════════════════════════════════════════════════
    # Phase 0: Bio-Inspired Scouting (Replaces legacy Global Pre-Scan)
    # ══════════════════════════════════════════════════════════════════
    # Step 0.1 (Saccade)  : Sample keyframes + DINOv2 saliency → hotspot map
    # Step 0.2 (Fixation) : PointBeam crops keyframes around hotspot ROI
    # Step 0.3 (Fovea)    : VLM reads cropped images → Multi-SKU intents
    # ══════════════════════════════════════════════════════════════════

    logger.info("Phase 0: Bio-Inspired Scouting...")
    prescan_frames = []
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    )  # pylint: disable=no-member
    prescan_interval = max(1, int(fps * 2))  # 1 frame every ~2 seconds
    temp_idx = 0
    while True:
        ret, f = cap.read()
        if not ret:
            break
        if temp_idx % prescan_interval == 0:
            f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member
            prescan_frames.append((temp_idx, f_rgb))
        temp_idx += 1

    # Reset capture to frame 0 for the main processing loop
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # pylint: disable=no-member

    logger.info(
        "[Scouting] Peripheral scan complete: %d keyframes from %d total (interval=%d)",
        len(prescan_frames),
        total_frames,
        prescan_interval,
    )

    # ── Step 0.1: Saccade — DINOv2 Saliency Aggregation ──────────
    # Aggregate saliency across sampled keyframes to find persistent
    # hotspots (areas that are consistently visually interesting).
    scouting_focus_roi = None
    foveated_intents = None
    scouting_latent_anchor = None

    try:
        from v2_logic.models.dinov2_engine import DINOv2Engine

        dinov2 = DINOv2Engine(device="cuda" if torch.cuda.is_available() else "cpu")

        # Accumulate saliency maps across keyframes
        aggregated_saliency = None
        sample_h, sample_w = None, None
        # Sample up to 8 evenly-spaced keyframes for saliency
        saliency_sample_indices = np.linspace(
            0, len(prescan_frames) - 1, min(8, len(prescan_frames)), dtype=int
        )

        for s_idx in saliency_sample_indices:
            _, frame_rgb = prescan_frames[s_idx]
            sample_h, sample_w = frame_rgb.shape[:2]
            sal_map = dinov2.generate_saliency_map(
                frame_rgb, original_size=(sample_h, sample_w)
            )
            if aggregated_saliency is None:
                aggregated_saliency = sal_map
            else:
                aggregated_saliency += sal_map

        # Normalize aggregated saliency
        if aggregated_saliency is not None:
            sal_max = aggregated_saliency.max()
            if sal_max > 0:
                aggregated_saliency /= sal_max

            logger.info(
                "[Scouting] Saccade: Aggregated saliency from %d keyframes. "
                "Mean=%.3f, Max=%.3f",
                len(saliency_sample_indices),
                aggregated_saliency.mean(),
                aggregated_saliency.max(),
            )

            # ── Step 0.2: Fixation — PointBeam ROI ────────────────
            hotspot_rois = dinov2.find_hotspot_roi(
                aggregated_saliency, top_k=1, min_hotspot_fraction=0.10
            )
            if hotspot_rois:
                scouting_focus_roi = hotspot_rois[0]  # (x1, y1, x2, y2)
                logger.info(
                    "[Scouting] PointBeam Fixation: ROI locked to %s",
                    scouting_focus_roi,
                )

                # --- LATENT LOCKING: Extract V-JEPA Anchor before VLM ---
                try:
                    from v2_logic.models.v_jepa_engine import VJEPAEngine

                    vjepa = VJEPAEngine(
                        device="cuda" if torch.cuda.is_available() else "cpu"
                    )

                    # Use the first valid prescan frame for encoding
                    _, first_prescan = prescan_frames[0]
                    fh, fw = first_prescan.shape[:2]

                    import torch.nn.functional as F

                    t_frame = (
                        torch.from_numpy(first_prescan)
                        .permute(2, 0, 1)
                        .float()
                        .unsqueeze(0)
                        / 255.0
                    )
                    t_frame_resized = F.interpolate(t_frame, size=(224, 224))

                    vjepa.encode(t_frame_resized)

                    x1, y1, x2, y2 = scouting_focus_roi
                    bbox_norm = {
                        "x": x1 / fw,
                        "y": y1 / fh,
                        "w": (x2 - x1) / fw,
                        "h": (y2 - y1) / fh,
                    }
                    scouting_latent_anchor = vjepa.extract_object_latent(bbox_norm)
                    logger.info(
                        "[Scouting] Latent Lock: Extracted Anchor (shape: %s)",
                        (
                            scouting_latent_anchor.shape
                            if scouting_latent_anchor is not None
                            else "None"
                        ),
                    )

                    del vjepa
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        import gc

                        gc.collect()
                except Exception as e:
                    logger.warning("[Scouting] Latent Anchor extraction failed: %s", e)

        # Clean up DINOv2 to free VRAM for VLM
        del dinov2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc

            gc.collect()

    except Exception as e:
        logger.warning("[Scouting] DINOv2 saliency failed: %s. Using full frame.", e)

    # ── Step 0.3: Fovea — VLM Focused Identification ─────────────
    # Crop keyframes around PointBeam ROI and ask VLM for brand labels.
    if scouting_focus_roi is not None:
        try:
            # V4.0: Unified VLM — Use SLMEngine (Qwen2.5-VL) for higher specificity
            from v2_logic.models.slm_engine import SLMEngine

            slm = SLMEngine()

            # Crop up to 4 representative keyframes around PointBeam ROI
            x1, y1, x2, y2 = scouting_focus_roi
            foveated_sample_indices = np.linspace(
                0, len(prescan_frames) - 1, min(4, len(prescan_frames)), dtype=int
            )
            all_discovered = []

            for f_idx in foveated_sample_indices:
                _, frame_rgb = prescan_frames[f_idx]
                fh, fw = frame_rgb.shape[:2]
                # Clamp ROI to frame bounds
                cx1 = max(0, min(x1, fw - 1))
                cy1 = max(0, min(y1, fh - 1))
                cx2 = max(cx1 + 1, min(x2, fw))
                cy2 = max(cy1 + 1, min(y2, fh))
                cropped = frame_rgb[cy1:cy2, cx1:cx2]

                # V4.0: Use Unified VLM discovery instead of legacy PaliGemma proxy
                discovered_dicts = slm.generate_initial_intents(
                    prompt=user_prompt, frame=cropped
                )

                for d in discovered_dicts:
                    if not d.get("is_contra"):
                        all_discovered.append(d["label"])

            # Deduplicate while preserving order
            seen = set()
            foveated_intents = []
            for intent_label in all_discovered:
                label_clean = intent_label.lower().strip()
                if label_clean not in seen:
                    seen.add(label_clean)
                    foveated_intents.append(intent_label)

            logger.info(
                "[Scouting] Unified Foveated Genesis: %d unique intents discovered: %s",
                len(foveated_intents),
                foveated_intents,
            )

            # Clean up SLM to free VRAM
            del slm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc

                gc.collect()

        except Exception as e:
            logger.warning(
                "[Scouting] Unified VLM foveated identification failed: %s", e
            )

    # ── Apply Scouting Results to Initial State ───────────────────
    scouting_updates = {"video_frames": prescan_frames}

    if foveated_intents:
        # Override seed intent with foveated discoveries
        target_intent = foveated_intents
        scouting_updates["active_intent"] = foveated_intents

        # Package into Genesis Intents with Latent Anchors
        genesis_intents = []
        for label in foveated_intents:
            gi = {
                "label": label,
                "confidence": 1.0,
                "bbox": (
                    {
                        "x": scouting_focus_roi[0],
                        "y": scouting_focus_roi[1],
                        "w": scouting_focus_roi[2] - scouting_focus_roi[0],
                        "h": scouting_focus_roi[3] - scouting_focus_roi[1],
                    }
                    if scouting_focus_roi
                    else {}
                ),
                "frame_idx": 0,
                "latent_pose": scouting_latent_anchor,
            }
            genesis_intents.append(gi)
        scouting_updates["genesis_intents"] = genesis_intents

        # Also seed active_latent_anchors for the first pass
        if scouting_latent_anchor is not None:
            scouting_updates["active_latent_anchors"] = [scouting_latent_anchor]

        logger.info(
            "[Scouting] Overriding seed intent with foveated: %s", foveated_intents
        )

    if scouting_focus_roi is not None:
        scouting_updates["focus_roi"] = scouting_focus_roi
        logger.info(
            "[Scouting] Injecting PointBeam ROI into initial state: %s",
            scouting_focus_roi,
        )

    initial_state["perception"] = initial_state["perception"].model_copy(
        update=scouting_updates
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 15 != 0:  # Process every 15 frames to save compute during tests
            frame_idx += 1
            continue

        logger.info(f"--- Processing Frame {frame_idx} ---")

        # FIX 1: Convert BGR (OpenCV default) to RGB (VLM/PIL expected format)
        # Without this, Red ↔ Blue channels are swapped, causing VLM color hallucinations
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member

        # Prepare state for current frame
        perception_update = initial_state["perception"].model_copy(
            update={
                "current_frame_idx": frame_idx,
                "image": frame_rgb,
                "last_frame": (
                    initial_state["perception"].image
                    if initial_state["perception"].image is not None
                    else frame
                ),
            }
        )
        initial_state["perception"] = perception_update

        # Invoke Graph
        try:
            # We pass the full state and let LangGraph manage the updates
            final_loop_state = app.invoke(initial_state)

            # Print summary results for the frame
            p = final_loop_state["perception"]
            d = final_loop_state["decision"]

            logger.info(f"Frame {frame_idx} Results:")
            logger.info(f"  - Count (N_visible): {p.n_visible}")
            logger.info(f"  - Volumetric Range: {p.n_volumetric_range}")
            logger.info(f"  - Spike Energy: {p.spike_energy:.2f}")
            logger.info(f"  - Anomaly Status: {d.status}")
            # V5.1: Log accumulated focus scores for Wait-and-Watch validation
            if p.latent_focus_scores:
                logger.info(
                    f"  - V5.1 Focus Scores: { {k: f'{v:.2f}' for k, v in p.latent_focus_scores.items()} }"
                )
            if d.slm_triggered:
                logger.info(f"  - SLM Reasoning: {d.slm_reasoning}")

            # Carry over state for the next frame (tracking, world model)
            # Some things should be reset per frame (like spikes), others persisted (tracked_objects)
            initial_state = final_loop_state

        except Exception as e:
            logger.error(f"Error in graph execution at frame {frame_idx}: {e}")
            import traceback

            traceback.print_exc()
            break

        frame_idx += 1

    cap.release()
    logger.info("Processing complete.")


if __name__ == "__main__":
    main()
