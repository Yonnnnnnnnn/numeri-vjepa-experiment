"""Glide-and-Count Integration Script (V2 Architecture)

Orchestrates the 4 layers:
1. v2e (Input)
2. VL-JEPA (Director)
3. V-JEPA (Brain)
4. CountGD (Executor)

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : Script (this file)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <Main>           → Script entry point                                    │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <Imports>        ← External dependencies                                 │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : str, int, etc.

Production Rules:
  Script          → Imports + Main
═══════════════════════════════════════════════════════════════════════════════

Pattern: Script
- Standalone executable for utility/testing purposes.
"""

import logging
import os
import sys

import cv2  # pylint: disable=no-member
import numpy as np  # pylint: disable=unused-import
import torch

# Add Implementation root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# pylint: disable=wrong-import-position
from v2_logic.models.v2e_engine import V2EEngine
from v2_logic.models.vl_jepa_engine import VLJEPAEngine
from v2_logic.models.v_jepa_engine import VJEPAEngine
from v2_logic.models.count_gd_engine import CountGDEngine

# pylint: enable=wrong-import-position

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlideCountPipeline:
    """
    Experimental asynchronous 4-layer pipeline for inventory counting.
    """

    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info("Initializing V2 Pipeline on %s...", self.device)

        self.layer1_input = V2EEngine(device=self.device)
        self.layer2_director = VLJEPAEngine(device=self.device)
        self.layer3_brain = VJEPAEngine(device=self.device)
        self.layer4_executor = CountGDEngine(device=self.device)

        self.temporal_counts = []
        self.frame_buffer = []

    def run_inference(self, video_path):
        """
        Run the full 4-layer inference loop on a video file.
        """
        if not os.path.exists(video_path):
            logger.error("Video not found: %s", video_path)
            return None

        cap = cv2.VideoCapture(video_path)  # pylint: disable=no-member
        fps = cap.get(cv2.CAP_PROP_FPS)  # pylint: disable=no-member
        frame_idx = 0

        # 1. Director identifies intent on first frame
        ret, first_frame = cap.read()
        if not ret:
            logger.error("Could not read video.")
            return None

        intent = self.layer2_director.identify_intent(first_frame)
        logger.info("[Director] Identified Intent: %s", intent)

        # 2. Main Glide Loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps

            # Layer 1: Generate Events (Low-level spike generation)
            _ = self.layer1_input.generate_events(frame, timestamp)

            # Layer 3 Buffer: V-JEPA expects 16 frames for temporal tokens
            rgb_frame = cv2.cvtColor(  # pylint: disable=no-member
                frame, cv2.COLOR_BGR2RGB  # pylint: disable=no-member
            )
            frame_tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float() / 255.0
            frame_tensor = torch.nn.functional.interpolate(
                frame_tensor.unsqueeze(0), size=(224, 224)
            ).squeeze(0)
            self.frame_buffer.append(frame_tensor)

            if len(self.frame_buffer) == 16:
                # Stack 16 frames: (1, 3, 16, 224, 224)
                # Note: V-JEPA ViT-L with num_frames=16 expects this temporal depth
                input_batch = (
                    torch.stack(self.frame_buffer).permute(1, 0, 2, 3).unsqueeze(0)
                )

                # Layer 3: Encode World State
                _ = self.layer3_brain.encode(input_batch)

                # Clear buffer (or slide it)
                self.frame_buffer = []

                # Layer 4: Precise Count (triggered every 16 frames for stable estimation)
                # Here we pass the last frame of the sequence + the latent context
                count = self.layer4_executor.count_frame(
                    input_batch[:, :, -1, :, :], prompt=intent
                )
                self.temporal_counts.append((timestamp, count))
                logger.debug("[Pipeline] Frame %d count: %d", frame_idx, count)

            frame_idx += 1

        # 3. Final Tally
        final_result = self.layer4_executor.tally_unique(self.temporal_counts)
        logger.info("Final Inventory Result: %d units of %s", final_result, intent)
        return final_result


if __name__ == "__main__":
    # Use one of the sample videos found
    sample_video = os.path.abspath(
        "Techs/sam2-main/sam2-main/demo/data/gallery/02_cups.mp4"
    )

    pipeline = GlideCountPipeline()
    pipeline.run_inference(sample_video)
