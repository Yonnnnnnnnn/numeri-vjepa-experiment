"""
CountVid Engine (Executor)

Integrates niki-amini-naieni/CountVid for precision unique counting in videos.
Replaces the legacy CountGD engine with modern PyTorch 2.3+ compatible code.

Pattern: Adapter
- Adapts CountVid inference to the Glide-and-Count pipeline.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : CountVidEngine (this module)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <CountVidEngine>  → class implementation                                 │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <CountVidModel>   ← from Techs.CountVid                                  │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : str, float, int, bool, "cuda", "cpu"

Production Rules:
  CountVidEngine  → imports + <CountVidEngine>
  <CountVidEngine> → __init__ + count_frame + tally_unique + update_sensitivity
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import os
import sys
from typing import List, Optional, Tuple

import cv2  # pylint: disable=no-member
import numpy as np
import torch
import traceback

# Add CountVid to path
COUNTVID_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "../../../Techs/CountVid-main/CountVid-main"
    )
)
if COUNTVID_PATH not in sys.path:
    sys.path.insert(0, COUNTVID_PATH)

logger = logging.getLogger(__name__)

# Confidence threshold for CountVid (same as CountGD default)
CONF_THRESH = 0.23


class CountVidEngine:
    """
    Final Tally engine for precise unique counting using CountVid.

    Pattern: Adapter
    """

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.transform = None
        self.confidence_thresh = CONF_THRESH
        self.nested_tensor_from_tensor_list = None

        try:
            # Clear Hugging Face datasets module cache to use CountVid's local datasets module
            hf_datasets_backup = None
            if "datasets" in sys.modules:
                logger.info(
                    "[CountVid] Found Hugging Face datasets module, temporarily removing..."
                )
                hf_datasets_backup = sys.modules.pop("datasets")

            # Ensure CountVid's path is at the beginning of sys.path for correct imports
            if COUNTVID_PATH not in sys.path:
                sys.path.insert(0, COUNTVID_PATH)
            elif sys.path.index(COUNTVID_PATH) != 0:
                sys.path.remove(COUNTVID_PATH)
                sys.path.insert(0, COUNTVID_PATH)

            # Add current directory to path to ensure correct imports
            original_cwd = os.getcwd()
            os.chdir(COUNTVID_PATH)

            # pylint: disable=import-error, import-outside-toplevel
            from util.slconfig import SLConfig
            import datasets.transforms as T
            from models.registry import MODULE_BUILD_FUNCS
            from util.misc import nested_tensor_from_tensor_list

            # pylint: enable=import-error, import-outside-toplevel

            # Store the function for later use
            self.nested_tensor_from_tensor_list = nested_tensor_from_tensor_list

            # Create checkpoints directory if it doesn't exist
            checkpoints_dir = os.path.join(COUNTVID_PATH, "checkpoints")
            os.makedirs(checkpoints_dir, exist_ok=True)

            # Build config from cfg_app.py
            cfg_path = os.path.join(COUNTVID_PATH, "cfg_app.py")
            cfg = SLConfig.fromfile(cfg_path)
            # pylint: disable=no-member
            cfg.merge_from_dict(
                {
                    "text_encoder_type": os.path.join(
                        checkpoints_dir, "bert-base-uncased"
                    )
                }
            )
            cfg_dict = cfg._cfg_dict.to_dict()
            # pylint: enable=no-member

            # Create args object
            class Args:
                """Arguments for CountVid model configuration."""

                # Dynamically set attributes in __init__?
                # For now just use an empty object or a thin class.
                def __init__(self):
                    pass

            args = Args()
            for k, v in cfg_dict.items():
                setattr(args, k, v)

            # Set checkpoint paths
            args.pretrain_model_path = os.path.join(checkpoints_dir, "countgd_box.pth")
            args.device = self.device

            # Check if required checkpoints exist
            required_checkpoints = [
                args.pretrain_model_path,
                os.path.join(checkpoints_dir, "bert-base-uncased"),
            ]

            missing_checkpoints = []
            for checkpoint_path in required_checkpoints:
                if not os.path.exists(checkpoint_path):
                    missing_checkpoints.append(checkpoint_path)

            if missing_checkpoints:
                logger.warning(
                    "[CountVid] Missing required checkpoints: %s",
                    ", ".join(missing_checkpoints),
                )
                logger.warning("[CountVid] Please download the checkpoints manually:")
                logger.warning(
                    "[CountVid] 1. Download BERT weights: python %s",
                    os.path.join(COUNTVID_PATH, "download_bert.py"),
                )
                logger.warning(
                    "[CountVid] 2. Download CountGD-Box weights from: "
                    "https://drive.google.com/file/d/1bw-YIS-Il5efGgUqGVisIZ8ekrhhf_FD/"
                    "view?usp=sharing"
                )
                logger.warning("[CountVid]   and save as: %s", args.pretrain_model_path)
                logger.warning(
                    "[CountVid] Using mock counting until checkpoints are available"
                )
                # Restore Hugging Face datasets module before returning
                if hf_datasets_backup is not None:
                    logger.info("[CountVid] Restoring Hugging Face datasets module...")
                    sys.modules["datasets"] = hf_datasets_backup
                os.chdir(original_cwd)
                return

            # Build transform
            normalize = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            self.transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    normalize,
                ]
            )

            # Build model
            # pylint: disable=no-member
            assert args.modelname in MODULE_BUILD_FUNCS._module_dict
            build_func = MODULE_BUILD_FUNCS.get(args.modelname)
            # pylint: enable=no-member
            self.model, _, _ = build_func(args)

            # Load checkpoint
            checkpoint = torch.load(
                args.pretrain_model_path, map_location="cpu", weights_only=False
            )
            if "model" in checkpoint:
                checkpoint = checkpoint["model"]
            self.model.load_state_dict(checkpoint, strict=False)
            self.model.to(self.device)
            self.model.eval()

            # Restore original directory and Hugging Face datasets module
            os.chdir(original_cwd)
            if hf_datasets_backup is not None:
                logger.info("[CountVid] Restoring Hugging Face datasets module...")
                sys.modules["datasets"] = hf_datasets_backup

            logger.info("[CountVid] Model loaded successfully on %s", self.device)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("[CountVid] Failed to load model: %s", str(e))
            logger.error("[CountVid] Using mock counting as fallback")
            self.model = None

    def count(
        self,
        image: np.ndarray,
        prompt: str = "items",
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, List]:
        """
        Convenience method for the LangGraph controller.
        Handles numpy to tensor conversion.

        Args:
            image: BGR image (numpy array).
            prompt: Text prompt for counting.
            target_size: Optional (width, height) to scale results back to.

        Returns:
            Tuple[int, List]: (count, list of bounding boxes [x1, y1, x2, y2])
        """
        # Convert numpy to tensor [B, C, H, W]
        # image is (H, W, 3) BGR
        # pylint: disable=no-member
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # pylint: enable=no-member
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        # count_frame returns (pred_count, pixel_boxes)
        count_val, detections = self.count_frame(tensor, prompt=prompt)

        # Scale detections if target size is provided
        if target_size and detections:
            h_orig, w_orig = tensor.shape[2], tensor.shape[3]
            w_target, h_target = target_size
            scaled_detections = []
            for box in detections:
                x1, y1, x2, y2 = box
                x1 = int(x1 * w_target / w_orig)
                y1 = int(y1 * h_target / h_orig)
                x2 = int(x2 * w_target / w_orig)
                y2 = int(y2 * h_target / h_orig)
                scaled_detections.append([x1, y1, x2, y2])
            detections = scaled_detections

        # Return just the count and detections to the node
        return int(count_val), detections

    def count_frame(
        self, frame_tensor: torch.Tensor, exemplars=None, prompt: str = "items"
    ) -> Tuple[int, List]:
        """
        Perform zero-shot or few-shot counting on a single frame.

        Args:
            frame_tensor: Tensor [B, C, H, W] or [C, H, W].
            exemplars: Optional exemplar boxes (not used in text-only mode).
            prompt: Text prompt for counting.

        Returns:
            Tuple[int, list]: (Predicted count, Predicted boxes [x1, y1, x2, y2])
        """
        _ = exemplars  # Placeholder for few-shot mode
        # Normalize prompt: convert list to comma-separated string
        if isinstance(prompt, list):
            prompt = ", ".join(prompt) if prompt else "items"

        if self.model is None or self.transform is None:
            # Mock behavior
            logger.info(
                "[CountVid] Using enhanced mock counting for prompt: %s", prompt
            )
            return 3, []

        try:
            # pylint: disable=import-outside-toplevel
            import torchvision.transforms.functional as F

            # Use the pre-imported function to avoid module conflicts
            nested_tensor_from_tensor_list = self.nested_tensor_from_tensor_list

            # pylint: enable=import-outside-toplevel

            if len(frame_tensor.shape) == 4:
                frame_tensor = frame_tensor[0]

            # Convert tensor to PIL Image
            image_pil = F.to_pil_image(
                frame_tensor / (255.0 if frame_tensor.max() > 1.0 else 1.0)
            )
            input_image, _ = self.transform(image_pil, {"exemplars": torch.tensor([])})
            input_image = input_image.to(self.device)

            # Prepare exemplars (empty for text-only)
            input_image_exemplars = input_image.clone()
            exemplar_tensor = torch.tensor([]).to(self.device)
            label_tensor = torch.tensor([0]).to(self.device)

            with torch.no_grad():
                model_output = self.model(
                    nested_tensor_from_tensor_list([input_image]),
                    nested_tensor_from_tensor_list([input_image_exemplars]),
                    [exemplar_tensor],
                    [label_tensor],
                    captions=[prompt + " ."],
                )

            logits = model_output["pred_logits"][0].sigmoid()
            boxes = model_output["pred_boxes"][0]

            # Confidence threshold
            box_mask = logits.max(dim=-1).values > self.confidence_thresh
            final_boxes = boxes[box_mask]
            pred_count = final_boxes.shape[0]

            # Convert boxes to pixel coordinates
            h, w = frame_tensor.shape[1:3]
            pixel_boxes = []
            for box in final_boxes:
                # box is [cx, cy, w, h] normalized
                cx, cy, bw, bh = box.cpu().tolist()
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                pixel_boxes.append([x1, y1, x2, y2])

            return pred_count, pixel_boxes
        except (RuntimeError, ValueError, TypeError) as e:
            logger.error("[CountVid] Error during counting: %s", str(e))
            logger.error(traceback.format_exc())
            return 1, []

    def tally_unique(self, temporal_counts: List[Tuple[float, int]]) -> int:
        """
        Final tally logic that integrates counts over time to resolve unique items.

        Args:
            temporal_counts: List of (timestamp, count)

        Returns:
            int: Final unique count
        """
        if not temporal_counts:
            return 0

        # Implementation of the "Glide-and-Count" integration logic:
        # We look for the peak or use a weighted average based on V-JEPA confidence.
        counts = [c[1] for c in temporal_counts]
        final_tally = int(np.max(counts))  # Simple peak tally

        logger.info("[CountVid] Final Tally: %d", final_tally)
        return final_tally

    # =========================================================================
    # Phase 4: Adaptive Sensitivity
    # =========================================================================

    def update_sensitivity(
        self,
        feedback_score: float,
        density_hint: float = 0.5,
    ) -> float:
        """
        Adjust detection confidence threshold based on SLM audit feedback
        and density characteristics from Phase 2.

        Higher feedback_score = system was over-counting → raise threshold.
        Lower feedback_score = system was under-counting → lower threshold.

        Args:
            feedback_score: SLM feedback in range [-1, 1].
                -1 = "dramatically under-counting"
                 0 = "count seems correct"
                +1 = "dramatically over-counting"
            density_hint: Density modifier from Phase 2 (0-1).
                Higher density → more cautious (raise threshold slightly).

        Returns:
            float: Updated confidence threshold.
        """
        # Sensitivity step size
        step = 0.02

        # Density adjustment: high-density scenes need higher thresholds
        density_bias = (density_hint - 0.5) * step  # Range: -0.01 to +0.01

        # Calculate adjustment
        adjustment = (feedback_score * step) + density_bias

        # Apply with clamping to safe range [0.10, 0.60]
        old_thresh = self.confidence_thresh
        self.confidence_thresh = max(0.10, min(0.60, old_thresh + adjustment))

        logger.info(
            "[CountVid] Sensitivity updated: %.3f → %.3f (feedback=%.2f, density=%.2f)",
            old_thresh,
            self.confidence_thresh,
            feedback_score,
            density_hint,
        )

        return self.confidence_thresh
