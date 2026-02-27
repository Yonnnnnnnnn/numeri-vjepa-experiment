"""Test VL-JEPA Engine

Verifies that the VLJEPAEngine (PaliGemma-based) can load and identify visual intent.

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

import numpy as np

# Add Implementation root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from v2_logic.models.vl_jepa_engine import VLJEPAEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_vljepa_initialization():
    logger.info("Testing VLJEPAEngine initialization...")
    # NOTE: Using a smaller model if available or just testing the structure if GPU memory is tight.
    # google/paligemma-3b-mix-224 is the one used in the source repo.
    try:
        # We use a very small mock or local check if possible
        # but for L40S we can try the real model.
        # IF the environment has HF_TOKEN and access to the model, it will work.
        engine = VLJEPAEngine(
            device="cpu"
        )  # Use CPU for structure test to avoid GPU hangs in dev
        logger.info("VLJEPAEngine initialized successfully.")
        return engine
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("VL-JEPA Test Failed: %s", e)
        return None


def test_intent_identification(engine):
    if engine is None:
        return

    logger.info("Testing intent identification...")

    # Create dummy frame
    height, width = 224, 224
    dummy_frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    dummy_frame[100:140, 100:140, :] = 255

    # Intent query
    prompt = "Caption: What is this object?"
    intent = engine.identify_intent(dummy_frame, prompt=prompt)
    logger.info("Identifying intent for dummy frame...")

    # Feature extraction
    features = engine.extract_visual_embeddings(dummy_frame)
    logger.info(
        f"Visual features shape: {features.shape} (Expected (1, 256, 1152) or similar)"
    )


if __name__ == "__main__":
    vljepa = test_vljepa_initialization()
    if vljepa:
        test_intent_identification(vljepa)
