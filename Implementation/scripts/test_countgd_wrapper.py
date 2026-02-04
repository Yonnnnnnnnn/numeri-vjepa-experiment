"""Test CountVid Engine

Verifies initialization and basic tally logic for CountVid.

Pattern: Script
- Standalone executable for utility/testing purposes.
"""

import logging
import os
import sys

import torch

# Add Implementation root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from v2_logic.models.count_vid_engine import CountVidEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_countvid():
    logger.info("Initializing CountVidEngine...")
    try:
        engine = CountVidEngine(device="cpu")
        logger.info("CountVidEngine initialized.")

        # Test basic frame counting
        dummy_frame = torch.randn(1, 3, 224, 224)
        count, detections = engine.count_frame(dummy_frame)
        logger.info(f"Frame count: {count}")
        logger.info(f"Detections: {detections}")

        # Test tally logic
        temporal_counts = [(1.0, 5), (2.0, 7), (3.0, 6)]
        final_tally = engine.tally_unique(temporal_counts)
        logger.info(f"Final Tally: {final_tally}")

        if final_tally == 7:
            logger.info("CountVid Layer verification PASSED.")
        else:
            logger.error(f"Unexpected tally: {final_tally}")

    except Exception as e:
        logger.error(f"CountVid test failed: {e}")


if __name__ == "__main__":
    test_countvid()
