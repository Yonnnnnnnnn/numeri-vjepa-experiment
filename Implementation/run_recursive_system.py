"""
Run Recursive Intent System

Main execution script for the end-to-end Recursive Intent system.
Orchestrates LangGraph execution on real video data.
"""

import os
import sys
import logging
import cv2
import time
import numpy as np

# Add Implementation root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

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

    # 1. Initialize Graph
    logger.info("Building Recursive Intent Graph...")
    app = create_recursive_flow_app()

    # 2. Input Source
    video_path = os.path.abspath(
        "Techs/sam2-main/sam2-main/demo/data/gallery/02_cups.mp4"
    )
    if not os.path.exists(video_path):
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
