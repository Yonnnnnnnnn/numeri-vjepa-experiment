"""Test V2E Engine

Verifies that the V2EEngine wrapper correctly initializes and generates events.

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

import numpy as np
import logging
import sys
import os

# Add Implementation root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from v2_logic.models.v2e_engine import V2EEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_v2e_initialization():
    logger.info("Testing V2EEngine initialization...")
    try:
        engine = V2EEngine(device="cpu")  # Use CPU for simple test to avoid CUDA issues
        logger.info("V2EEngine initialized successfully.")
        return engine
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Failed to initialize V2EEngine: %s", e)
        return None


def test_event_generation(engine):
    if engine is None:
        return

    logger.info("Testing event generation...")

    # Create a synthetic sequence: a moving white square on black background
    H, W = 240, 320

    # Frame 0: Black
    frame0 = np.zeros((H, W), dtype=np.uint8)
    events0 = engine.generate_events(frame0, 0.0)
    logger.info(
        "Frame 0 (0.0s): Generated %d events (Expected 0 on first frame)", len(events0)
    )

    # Frame 1: White square in middle
    frame1 = np.zeros((H, W), dtype=np.uint8)
    frame1[100:140, 140:180] = 255
    events1 = engine.generate_events(frame1, 0.1)
    logger.info(
        "Frame 1 (0.1s): Generated %d events (Expected many ON events)", len(events1)
    )

    if len(events1) > 0:
        logger.info("Sample event: %s", events1[0])
        # Check polarities
        on_events = np.sum(events1[:, 3] == 1)
        off_events = np.sum(events1[:, 3] == -1)
        logger.info("ON events: %d, OFF events: %d", on_events, off_events)

    # Frame 2: Square moved
    frame2 = np.zeros((H, W), dtype=np.uint8)
    frame2[110:150, 150:190] = 255
    events2 = engine.generate_events(frame2, 0.2)
    logger.info(
        "Frame 2 (0.2s): Generated %d events (Expected ON and OFF events)", len(events2)
    )

    if len(events2) > 0:
        on_events = np.sum(events2[:, 3] == 1)
        off_events = np.sum(events2[:, 3] == -1)
        logger.info("ON events: %d, OFF events: %d", on_events, off_events)


if __name__ == "__main__":
    v2e_engine = test_v2e_initialization()
    if v2e_engine:
        test_event_generation(v2e_engine)
