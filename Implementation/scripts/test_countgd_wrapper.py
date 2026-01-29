"""Test CountGD Engine

Verifies initialization and basic tally logic for Layer 4.

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

import torch

# Add Implementation root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from v2_logic.models.count_gd_engine import CountGDEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_countgd():
    logger.info("Initializing CountGDEngine...")
    try:
        engine = CountGDEngine(device="cpu")
        logger.info("CountGDEngine initialized.")

        # Test basic frame counting
        dummy_frame = torch.randn(1, 3, 224, 224)
        count = engine.count_frame(dummy_frame)
        logger.info(f"Frame count: {count}")

        # Test tally logic
        temporal_counts = [(1.0, 5), (2.0, 7), (3.0, 6)]
        final_tally = engine.tally_unique(temporal_counts)
        logger.info(f"Final Tally: {final_tally}")

        if final_tally == 7:
            logger.info("CountGD Layer 4 verification PASSED.")
        else:
            logger.error(f"Unexpected tally: {final_tally}")

    except Exception as e:
        logger.error(f"CountGD test failed: {e}")


if __name__ == "__main__":
    test_countgd()
