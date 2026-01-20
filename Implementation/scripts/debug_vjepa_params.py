"""
Debug script to inspect V-JEPA predictor parameters and position embeddings.
"""

import os
import sys

# Setup paths (matches v_jepa_engine.py)
JEPA_ROOT = os.path.abspath("Techs/jepa-main/jepa-main")
JEPA_SRC = os.path.join(JEPA_ROOT, "src")
if JEPA_SRC not in sys.path:
    sys.path.insert(0, JEPA_SRC)
if JEPA_ROOT not in sys.path:
    sys.path.insert(0, JEPA_ROOT)

try:
    from models.predictor import vit_predictor  # pylint: disable=import-error

    print(f"Imported vit_predictor from: {sys.modules['models.predictor'].__file__}")

    model = vit_predictor(
        embed_dim=1024,
        predictor_embed_dim=384,
        num_frames=16,
        tubelet_size=2,
        depth=6,
        num_heads=16,
    )
    print(f"Model num_frames: {model.num_frames}")
    print(f"Model is_video: {model.is_video}")
    print(f"Model num_patches: {model.num_patches}")
    print(f"Model predictor_pos_embed shape: {model.predictor_pos_embed.shape}")

except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"Error: {e}")
