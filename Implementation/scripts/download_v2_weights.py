"""Download Model Weights

Fetches weights for all V2 layers:
- Layer 1: Done (v2e)
- Layer 2: via VLJEPAEngine (Transformers)
- Layer 3: V-JEPA (Meta FAIR)
- Layer 4: CountGD

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

import os
import urllib.request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Target paths
# Use relative path for portability (e.g. for Google Colab)
CHECKPOINT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../checkpoints")
)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

WEIGHTS = {
    # V-JEPA ViT-L/16 (Layer 3)
    "vjepa_vitl16": {
        "url": "https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar",
        "path": os.path.join(CHECKPOINT_DIR, "vjepa_vitl16.pth.tar"),
    },
    # CountGD (GroundingDINO-X/SAM, Layer 4)
    # Placeholder: CountGD usually requires custom weight mapping.
}


def download_file(url, path):
    if os.path.exists(path):
        logger.info(f"File already exists: {path}")
        return

    logger.info(f"Downloading {url} to {path}...")
    try:
        urllib.request.urlretrieve(url, path)
        logger.info("Download complete.")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")


if __name__ == "__main__":
    for name, info in WEIGHTS.items():
        download_file(info["url"], info["path"])
