"""
Segmentation Engine Module

Uses SAM2 (Segment Anything Model 2) for automatic instance segmentation.
Segments individual objects in frames for per-item counting.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : SegmentationEngine (this file)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <SegmentationEngine>  → Main wrapper for SAM2 segmentation              │
  │  <SegmentResult>       → Output dataclass with masks and boxes           │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <SAM2ImagePredictor>   ← from sam2_src (SAM2 model)                      │
  │  <build_sam2>           ← from sam2_src.build_sam (model builder)         │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : numpy.ndarray, torch.Tensor, list, dict

Production Rules:
  SegmentationEngine → __init__ + segment_frame
  segment_frame → preprocess + SAM2 forward + postprocess → SegmentResult
═══════════════════════════════════════════════════════════════════════════════

Pattern: Adapter
- Adapts SAM2's complex API to a simple interface
- Hides model loading and preprocessing details
"""

import sys
import os
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Add sam2_src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass
class SegmentResult:
    """Result of segmentation on a single frame."""

    masks: List[np.ndarray]  # List of binary masks (H, W)
    boxes: List[Tuple[int, int, int, int]]  # List of (x1, y1, x2, y2) bounding boxes
    scores: List[float]  # Confidence scores
    num_segments: int  # Total number of segments

    def get_cropped_regions(
        self, image: np.ndarray, padding: int = 5
    ) -> List[np.ndarray]:
        """Extract cropped regions for each segment."""
        crops = []
        h, w = image.shape[:2]
        for box in self.boxes:
            x1, y1, x2, y2 = box
            # Add padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            crops.append(image[y1:y2, x1:x2])
        return crops


class SegmentationEngine:
    """
    SAM2-based segmentation engine for inventory counting.
    Uses automatic mask generation to segment all objects in a frame.
    """

    def __init__(
        self,
        model_cfg: str = "sam2_hiera_t.yaml",  # Tiny model for T4 compatibility
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        min_mask_area: int = 500,  # Minimum mask area (filter small/far objects)
        max_mask_area: int = 100000,  # Maximum mask area (filter background)
        score_threshold: float = 0.7,
        roi_margin: float = 0.15,  # Ignore 15% edge of frame (focus on center)
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.min_mask_area = min_mask_area
        self.max_mask_area = max_mask_area
        self.score_threshold = score_threshold
        self.roi_margin = roi_margin  # 0.15 = focus on center 70% of frame
        self.model = None
        self.predictor = None

        # Try to load SAM2
        self._load_model(model_cfg, checkpoint_path)

    def _load_model(self, model_cfg: str, checkpoint_path: Optional[str]):
        """Load SAM2 model with fallback to HuggingFace."""
        try:
            from sam2.build_sam import build_sam2, build_sam2_hf
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            print("[SegmentationEngine] Loading SAM2...")

            model_loaded = False

            # Try loading from local checkpoint first
            if checkpoint_path and os.path.exists(checkpoint_path):
                ckpt_name = os.path.basename(checkpoint_path).lower()
                print(f"[SegmentationEngine] Found checkpoint: {ckpt_name}")

                # Auto-detect HuggingFace model ID from checkpoint name
                if "2.1" in ckpt_name or "sam2.1" in ckpt_name:
                    if "tiny" in ckpt_name:
                        hf_model_id = "facebook/sam2.1-hiera-tiny"
                        config_name = "configs/sam2.1/sam2.1_hiera_t.yaml"
                    elif "small" in ckpt_name:
                        hf_model_id = "facebook/sam2.1-hiera-small"
                        config_name = "configs/sam2.1/sam2.1_hiera_s.yaml"
                    elif "base" in ckpt_name:
                        hf_model_id = "facebook/sam2.1-hiera-base-plus"
                        config_name = "configs/sam2.1/sam2.1_hiera_b+.yaml"
                    elif "large" in ckpt_name:
                        hf_model_id = "facebook/sam2.1-hiera-large"
                        config_name = "configs/sam2.1/sam2.1_hiera_l.yaml"
                    else:
                        hf_model_id = "facebook/sam2.1-hiera-tiny"
                        config_name = "configs/sam2.1/sam2.1_hiera_t.yaml"
                else:
                    if "tiny" in ckpt_name:
                        hf_model_id = "facebook/sam2-hiera-tiny"
                        config_name = "configs/sam2/sam2_hiera_t.yaml"
                    elif "small" in ckpt_name:
                        hf_model_id = "facebook/sam2-hiera-small"
                        config_name = "configs/sam2/sam2_hiera_s.yaml"
                    elif "base" in ckpt_name:
                        hf_model_id = "facebook/sam2-hiera-base-plus"
                        config_name = "configs/sam2/sam2_hiera_b+.yaml"
                    elif "large" in ckpt_name:
                        hf_model_id = "facebook/sam2-hiera-large"
                        config_name = "configs/sam2/sam2_hiera_l.yaml"
                    else:
                        hf_model_id = "facebook/sam2-hiera-tiny"
                        config_name = "configs/sam2/sam2_hiera_t.yaml"

                # Try local build first
                try:
                    print(f"[SegmentationEngine] Using config: {config_name}")
                    self.model = build_sam2(
                        config_name, checkpoint_path, device=self.device
                    )
                    model_loaded = True
                    print("[SegmentationEngine] Loaded from local checkpoint.")
                except Exception as local_err:
                    print(f"[SegmentationEngine] Local load failed: {local_err}")
                    print("[SegmentationEngine] Trying HuggingFace fallback...")

                    # Fallback to HuggingFace (will download if needed)
                    try:
                        self.model = build_sam2_hf(hf_model_id, device=self.device)
                        model_loaded = True
                        print(
                            f"[SegmentationEngine] Loaded from HuggingFace: {hf_model_id}"
                        )
                    except Exception as hf_err:
                        print(
                            f"[SegmentationEngine] HuggingFace load also failed: {hf_err}"
                        )

            # If no local checkpoint, try HuggingFace directly
            if not model_loaded:
                print(
                    "[SegmentationEngine] No local checkpoint. Loading from HuggingFace..."
                )
                try:
                    hf_model_id = "facebook/sam2.1-hiera-tiny"  # Default to tiny
                    self.model = build_sam2_hf(hf_model_id, device=self.device)
                    model_loaded = True
                    print(
                        f"[SegmentationEngine] Loaded from HuggingFace: {hf_model_id}"
                    )
                except Exception as hf_err:
                    print(f"[SegmentationEngine] HuggingFace load failed: {hf_err}")
                    print("[SegmentationEngine] SAM2 not available.")
                    return

            # Create automatic mask generator
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=self.model,
                points_per_side=16,  # Reduced for speed
                pred_iou_thresh=0.8,
                stability_score_thresh=0.85,
                min_mask_region_area=self.min_mask_area,
            )

            print(f"[SegmentationEngine] SAM2 loaded on {self.device}")

        except ImportError as e:
            print(f"[SegmentationEngine] Error loading SAM2: {e}")
            print("[SegmentationEngine] Falling back to simple grid segmentation.")
            self.mask_generator = None

    def segment_frame(self, image: np.ndarray) -> SegmentResult:
        """
        Segment all objects in a frame with distance-based filtering.

        Args:
            image: RGB image as numpy array (H, W, 3)

        Returns:
            SegmentResult with masks, boxes, and scores (filtered by area and ROI)
        """
        if self.mask_generator is None:
            # Fallback: Return empty result
            return SegmentResult(masks=[], boxes=[], scores=[], num_segments=0)

        # Run SAM2 automatic mask generator
        try:
            masks_data = self.mask_generator.generate(image)
        except Exception as e:
            print(f"[SegmentationEngine] Error during segmentation: {e}")
            return SegmentResult(masks=[], boxes=[], scores=[], num_segments=0)

        # Calculate ROI bounds (center region of frame)
        h, w = image.shape[:2]
        roi_x1 = int(w * self.roi_margin)
        roi_y1 = int(h * self.roi_margin)
        roi_x2 = int(w * (1 - self.roi_margin))
        roi_y2 = int(h * (1 - self.roi_margin))

        # Extract masks, boxes, scores with filtering
        masks = []
        boxes = []
        scores = []
        filtered_count = 0

        for mask_info in masks_data:
            # Filter by score
            if mask_info.get("predicted_iou", 0) < self.score_threshold:
                continue

            # Get mask and calculate area
            mask = mask_info["segmentation"]
            mask_area = mask_info.get("area", np.sum(mask))

            # Filter by area range (distance proxy)
            if mask_area < self.min_mask_area:
                filtered_count += 1
                continue  # Too small = too far
            if mask_area > self.max_mask_area:
                filtered_count += 1
                continue  # Too large = background/container

            # Get bounding box
            bbox = mask_info["bbox"]  # (x, y, w, h) format
            x, y, bw, bh = bbox
            box_x1, box_y1 = int(x), int(y)
            box_x2, box_y2 = int(x + bw), int(y + bh)

            # Calculate box center
            box_cx = (box_x1 + box_x2) // 2
            box_cy = (box_y1 + box_y2) // 2

            # Filter by ROI (center of box must be within ROI)
            if not (roi_x1 <= box_cx <= roi_x2 and roi_y1 <= box_cy <= roi_y2):
                filtered_count += 1
                continue  # Outside focus area

            masks.append(mask)
            boxes.append((box_x1, box_y1, box_x2, box_y2))
            scores.append(mask_info.get("predicted_iou", 0.0))

        if filtered_count > 0:
            print(
                f"[SegmentationEngine] Filtered {filtered_count} objects (outside ROI/area range)"
            )

        return SegmentResult(
            masks=masks, boxes=boxes, scores=scores, num_segments=len(masks)
        )

    def visualize_masks(
        self, image: np.ndarray, result: SegmentResult, alpha: float = 0.5
    ) -> np.ndarray:
        """Overlay masks on image for visualization."""
        output = image.copy()

        # Generate random colors for each mask
        colors = [
            tuple(np.random.randint(0, 255, 3).tolist())
            for _ in range(result.num_segments)
        ]

        for mask, color in zip(result.masks, colors):
            # Create colored overlay
            colored_mask = np.zeros_like(image)
            colored_mask[mask] = color
            output = (output * (1 - alpha) + colored_mask * alpha).astype(np.uint8)

        return output


if __name__ == "__main__":
    # Quick test
    engine = SegmentationEngine()
    print("Segmentation engine initialized.")
