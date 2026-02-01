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
    os.path.join(PARENT_DIR, "Techs/CountGD-main/CountGD-main"),
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
    target_intent = ["cup"]
    initial_state = create_initial_state(session_id, target_intent, time.time())

    logger.info(f"Session started: {session_id}")
    logger.info(f"Target Intent: {target_intent}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 15 != 0:  # Process every 15 frames to save compute during tests
            frame_idx += 1
            continue

        logger.info(f"--- Processing Frame {frame_idx} ---")

        # Prepare state for current frame
        perception_update = initial_state["perception"].model_copy(
            update={
                "current_frame_idx": frame_idx,
                "image": frame,
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
