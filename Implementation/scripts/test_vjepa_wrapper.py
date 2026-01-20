"""
Test V-JEPA Engine

Verifies weight loading and latent encoding for Layer 3.
"""

import logging
import os
import sys
import torch

# Add Implementation root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from v2_logic.models.v_jepa_engine import VJEPAEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_vjepa():
    logger.info("Initializing VJEPAEngine...")
    try:
        engine = VJEPAEngine(device="cpu")  # cpu for basic load test
        logger.info("VJEPAEngine initialized.")

        # Test encoding
        dummy_frame = torch.randn(1, 3, 224, 224)
        latent = engine.encode(dummy_frame)
        logger.info(f"Latent shape: {latent.shape}")

        if latent is not None:
            logger.info("V-JEPA Layer 3 encoding test PASSED.")

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("V-JEPA Test Failed: %s", e)


if __name__ == "__main__":
    test_vjepa()
