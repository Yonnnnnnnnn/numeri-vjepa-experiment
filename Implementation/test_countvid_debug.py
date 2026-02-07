"""
Test CountVid Debug

Simple script to reproduce the CountVid runtime error.
"""

import os
import sys
import torch
import cv2
import numpy as np

# Add Implementation root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Robustly add Techs dependencies
TECHS_PATHS = [
    os.path.join(PARENT_DIR, "Techs/Depth-Anything-V2-main/Depth-Anything-V2-main"),
    os.path.join(PARENT_DIR, "Techs/CountVid-main/CountVid-main"),
    os.path.join(PARENT_DIR, "Techs/sam2-main/sam2-main"),
    os.path.join(PARENT_DIR, "Techs/v2e-master/v2e-master"),
]

for p in TECHS_PATHS:
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)

from v2_logic.models.count_vid_engine import CountVidEngine


def main():
    print("Initializing CountVidEngine...")
    try:
        engine = CountVidEngine(device="cuda" if torch.cuda.is_available() else "cpu")

        # Create a dummy image
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        print("Running count_frame...")
        result = engine.count_frame(img, ["person"], 0)
        print(f"Result: {result}")

    except Exception as e:
        print(f"Caught exception: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
