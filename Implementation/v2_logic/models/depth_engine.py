"""
Depth Estimation Engine (DepthAnything V2)

Wraps DepthAnything V2 for monocular depth estimation.
Provides relative depth maps for 3D point cloud back-projection.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : DepthEngine (this module)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <DepthEngine>    → Main wrapper for DepthAnything V2                     │
  │  <DepthResult>    → Output dataclass with depth map and stats             │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <DepthAnythingV2>  ← from depth_anything_v2.dpt (Model)                  │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : numpy.ndarray, torch.Tensor, float, int

Production Rules:
  DepthEngine      → __init__ + estimate_depth
  estimate_depth   → preprocess + model.infer_image + postprocess → DepthResult
═══════════════════════════════════════════════════════════════════════════════

Pattern: Adapter
- Adapts DepthAnything V2's API to a simple interface for the Recursive Intent system.
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Add DepthAnything V2 to path
DEPTH_ANYTHING_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../Techs/Depth-Anything-V2-main/Depth-Anything-V2-main",
    )
)
if DEPTH_ANYTHING_PATH not in sys.path:
    sys.path.insert(0, DEPTH_ANYTHING_PATH)

# Ensure depth_anything_v2 is a valid package (fix for missing __init__.py in some environments)
_pkg_init = os.path.join(DEPTH_ANYTHING_PATH, "depth_anything_v2/__init__.py")
if os.path.exists(os.path.dirname(_pkg_init)) and not os.path.exists(_pkg_init):
    try:
        with open(_pkg_init, "w") as f:
            f.write("# Auto-generated package init\n")
    except Exception:
        pass


@dataclass
class DepthResult:
    """Result of depth estimation on a single frame."""

    depth_map: np.ndarray  # Relative depth map (H, W), values 0-1
    raw_depth: np.ndarray  # Raw model output before normalization
    stats: Dict[str, float]  # Statistics: min, max, mean depth
    has_depth: bool  # Whether depth estimation succeeded


class DepthEngine:
    """
    DepthAnything V2 wrapper for relative depth estimation.

    Pattern: Adapter
    """

    # Model configurations for different encoder sizes
    MODEL_CONFIGS = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
    }

    def __init__(
        self,
        encoder: str = "vits",  # Use smallest model for T4 compatibility
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        input_size: int = 518,
    ):
        """
        Initialize DepthAnything V2.

        Args:
            encoder: Model encoder size ('vits', 'vitb', 'vitl').
            checkpoint_path: Path to model checkpoint. If None, uses default.
            device: Device to run model on.
            input_size: Input image size for inference.
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.encoder = encoder
        self.input_size = input_size
        self.model = None

        # Default checkpoint path
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                DEPTH_ANYTHING_PATH, f"checkpoints/depth_anything_v2_{encoder}.pth"
            )

        self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str) -> None:
        """Load DepthAnything V2 model."""
        try:
            from depth_anything_v2.dpt import DepthAnythingV2

            if self.encoder not in self.MODEL_CONFIGS:
                logger.warning(
                    "[DepthEngine] Unknown encoder %s, using 'vits'", self.encoder
                )
                self.encoder = "vits"

            config = self.MODEL_CONFIGS[self.encoder]

            if os.path.exists(checkpoint_path):
                logger.info(
                    "[DepthEngine] Loading from checkpoint: %s", checkpoint_path
                )
                self.model = DepthAnythingV2(**config)
                self.model.load_state_dict(
                    torch.load(checkpoint_path, map_location="cpu")
                )
                self.model = self.model.to(self.device).eval()
                logger.info(
                    "[DepthEngine] Model loaded on %s (encoder=%s)",
                    self.device,
                    self.encoder,
                )
            else:
                logger.warning(
                    "[DepthEngine] Checkpoint not found: %s", checkpoint_path
                )
                logger.warning(
                    "[DepthEngine] Please download from HuggingFace: "
                    "depth-anything/Depth-Anything-V2-Small"
                )
                self.model = None

        except ImportError as e:
            logger.error("[DepthEngine] Failed to import DepthAnythingV2: %s", e)
            self.model = None
        except Exception as e:
            logger.error("[DepthEngine] Error loading model: %s", e)
            self.model = None

    def estimate_depth(self, image: np.ndarray) -> DepthResult:
        """
        Estimate relative depth from an RGB image.

        Args:
            image: RGB image as numpy array (H, W, 3), BGR or RGB.

        Returns:
            DepthResult with depth map and statistics.
        """
        if self.model is None:
            logger.warning("[DepthEngine] Model not loaded, returning mock depth")
            h, w = image.shape[:2]
            mock_depth = np.zeros((h, w), dtype=np.float32)
            return DepthResult(
                depth_map=mock_depth,
                raw_depth=mock_depth,
                stats={"min": 0.0, "max": 0.0, "mean": 0.0},
                has_depth=False,
            )

        try:
            # DepthAnything expects BGR input (cv2 format)
            # If RGB, no conversion needed as the model handles it internally

            # Run inference
            with torch.no_grad():
                raw_depth = self.model.infer_image(image, self.input_size)

            # Normalize to 0-1 range
            depth_min = raw_depth.min()
            depth_max = raw_depth.max()
            if depth_max - depth_min > 0:
                depth_normalized = (raw_depth - depth_min) / (depth_max - depth_min)
            else:
                depth_normalized = np.zeros_like(raw_depth)

            # Calculate statistics
            stats = {
                "min": float(depth_min),
                "max": float(depth_max),
                "mean": float(raw_depth.mean()),
            }

            return DepthResult(
                depth_map=depth_normalized.astype(np.float32),
                raw_depth=raw_depth.astype(np.float32),
                stats=stats,
                has_depth=True,
            )

        except Exception as e:
            logger.error("[DepthEngine] Inference error: %s", e)
            h, w = image.shape[:2]
            mock_depth = np.zeros((h, w), dtype=np.float32)
            return DepthResult(
                depth_map=mock_depth,
                raw_depth=mock_depth,
                stats={"min": 0.0, "max": 0.0, "mean": 0.0},
                has_depth=False,
            )

    def visualize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Convert a depth map to a colorized BGR image for display.

        Args:
            depth_map: Normalized depth map (0-1).

        Returns:
            Colorized BGR image (H, W, 3).
        """
        # pylint: disable=no-member
        import cv2

        depth_8bit = (depth_map * 255).astype(np.uint8)
        color_depth = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_MAGMA)
        return color_depth


if __name__ == "__main__":
    # Quick test
    import cv2

    engine = DepthEngine(encoder="vits")

    # Create a dummy image for testing
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = engine.estimate_depth(dummy_image)

    print(f"Depth map shape: {result.depth_map.shape}")
    print(f"Stats: {result.stats}")
    print(f"Has depth: {result.has_depth}")
