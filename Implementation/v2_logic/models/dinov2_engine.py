"""
DINOv2 Engine (Phase 2)

Wraps Meta's DINOv2 (ViT-B/14) for robust feature extraction, "visual chaos" analysis,
and saliency map generation for Bio-Inspired Scouting (PointBeam Focus).
This engine extracts a (768,) semantic vector for density regression, analyzes
feature map variance as a proxy for physical complexity/specularity, and generates
spatial saliency maps from patch token norms to guide the Foveated Interaction system.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : DINOv2Engine (this module)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <DINOv2Engine>  → __init__ | extract_features | analyze_specularity      │
  │                    | generate_saliency_map | find_hotspot_roi             │
  │  <Helper>        → _preprocess                                            │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <torch>         ← from torch (tensor ops)                                │
  │  <torch.hub>     ← from torch (model loading)                             │
  │  <transforms>    ← from torchvision (image preprocessing)                 │
  │  <Image>         ← from PIL (image loading)                               │
  │  <cv2>           ← from cv2 (image resizing for saliency upscale)         │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : "cuda", "cpu", "facebookresearch/dinov2", "dinov2_vitb14"

Production Rules:
  DINOv2Engine    → imports + <DINOv2Engine>
  <DINOv2Engine>  → class DINOv2Engine: <Methods>+
  <Methods>       → __init__(device)
                  | extract_features(image) -> Tensor(768)
                  | analyze_specularity(image) -> float (variance score)
                  | generate_saliency_map(image) -> np.ndarray (H, W) heatmap
                  | find_hotspot_roi(saliency_map, top_k) -> List[Tuple] ROIs
═══════════════════════════════════════════════════════════════════════════════
"""

import logging

import cv2  # pylint: disable=no-member
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


class DINOv2Engine:
    """
    Wraps DINOv2 model for feature extraction and visual complexity analysis.

    Pattern: Adapter
    - Adapts the raw DINOv2 model interface to our specific needs (extract specs, density features).
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize DINOv2 engine.
        Loads 'dinov2_vitb14' from torch hub.
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"[DINOv2Engine] Loading DINOv2 ViT-B/14 on {self.device}...")

        # Load DINOv2 model
        try:
            self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(
                self.device
            )
            self.model.eval()
        except Exception as e:
            print(f"[DINOv2Engine] Error loading model: {e}")
            raise e

        # Standard ImageNet transform
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def extract_features(self, image) -> torch.Tensor:
        """
        Extracts the globally pooled feature vector (CLS token) from an image.

        Args:
            image (PIL.Image or Tensor): Input image.

        Returns:
            torch.Tensor: (768,) feature vector.
        """
        img_tensor = self._preprocess(image)

        with torch.no_grad():
            # efficient forward pass
            features = self.model(img_tensor)
            # DINOv2 output is already (batch, 768) for the CLS token
            return features.squeeze(0).cpu()  # Return (768,)

    def analyze_specularity(self, image) -> float:
        """
        Analyzes 'visual chaos' or specularity/complexity by looking at the
        variance of the patch definitions from the last layer.

        This serves as a heuristic for physical density:
        Higher variance in feature map often correlates with higher texture complexity.

        Args:
            image (PIL.Image or Tensor): Input image.

        Returns:
            float: Complexity score (variance * 1000 for readability).
        """
        img_tensor = self._preprocess(image)

        with torch.no_grad():
            # Get intermediate layers or patch tokens?
            # Standard forward() returns the CLS token. We need patch tokens.
            # DINOv2 standard forward_features method returns dict with 'x_norm_patchtokens'

            output = self.model.forward_features(img_tensor)
            patch_tokens = output["x_norm_patchtokens"]  # (1, N_patches, 768)

            # Calculate variance across patches
            # High variance means patches are very different -> high complexity/texture
            # Low variance means patches are similar -> smooth/uniform surface
            variance = torch.var(patch_tokens, dim=1).mean().item()

        return variance * 1000.0

    def generate_saliency_map(self, image, original_size: tuple = None) -> np.ndarray:
        """
        Generate a spatial saliency map from DINOv2 patch token norms.

        Each patch token's L2 norm indicates how "visually interesting" that
        spatial region is. High-norm patches correlate with textured, complex,
        or object-dense areas — ideal targets for PointBeam Focus.

        Args:
            image: PIL.Image, np.ndarray (H, W, 3), or torch.Tensor.
            original_size: Optional (H, W) to upscale the saliency map back
                           to the original image resolution.

        Returns:
            np.ndarray: Saliency map normalized to [0, 1].
                        Shape is (16, 16) if no original_size, else (H, W).
        """
        img_tensor = self._preprocess(image)

        with torch.no_grad():
            output = self.model.forward_features(img_tensor)
            patch_tokens = output["x_norm_patchtokens"]  # (1, N_patches, 768)

            # DINOv2 ViT-B/14 with 224px input: 224/14 = 16 patches per side
            num_patches_side = int(patch_tokens.shape[1] ** 0.5)

            # Compute per-patch L2 norm as saliency score
            patch_norms = torch.norm(patch_tokens, dim=-1)  # (1, N_patches)

            # Reshape to spatial grid (16, 16)
            saliency = (
                patch_norms.view(num_patches_side, num_patches_side).cpu().numpy()
            )

            # Normalize to [0, 1]
            saliency_min = saliency.min()
            saliency_max = saliency.max()
            if saliency_max > saliency_min:
                saliency = (saliency - saliency_min) / (saliency_max - saliency_min)
            else:
                saliency = np.zeros_like(saliency)

        # Upscale to original image resolution if requested
        if original_size is not None:
            target_h, target_w = original_size[:2]
            saliency = cv2.resize(
                saliency,
                (target_w, target_h),
                interpolation=cv2.INTER_CUBIC,
            )

        logger.info(
            "[DINOv2] Saliency map generated: shape=%s, max=%.3f, mean=%.3f",
            saliency.shape,
            saliency.max(),
            saliency.mean(),
        )
        return saliency

    def find_hotspot_roi(
        self,
        saliency_map: np.ndarray,
        top_k: int = 1,
        min_hotspot_fraction: float = 0.15,
    ) -> list:
        """
        Identify the top-K hotspot bounding boxes from a saliency map.

        Uses thresholding + connected components to find contiguous regions
        of high saliency, then returns their bounding boxes sorted by area.

        Args:
            saliency_map: np.ndarray (H, W) normalized [0, 1].
            top_k: Number of hotspot ROIs to return.
            min_hotspot_fraction: Minimum fraction of image area for a valid
                                  hotspot (avoids tiny noise regions).

        Returns:
            List of (x1, y1, x2, y2) bounding boxes in pixel coordinates.
        """
        h, w = saliency_map.shape[:2]
        min_area = int(h * w * min_hotspot_fraction)

        # Threshold at top 30% of saliency values
        threshold = np.percentile(saliency_map, 70)
        binary = (saliency_map >= threshold).astype(np.uint8) * 255

        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )

        # Collect valid hotspot regions (skip label 0 = background)
        hotspots = []
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < min_area:
                continue

            x1 = stats[label_id, cv2.CC_STAT_LEFT]
            y1 = stats[label_id, cv2.CC_STAT_TOP]
            bw = stats[label_id, cv2.CC_STAT_WIDTH]
            bh = stats[label_id, cv2.CC_STAT_HEIGHT]
            hotspots.append((x1, y1, x1 + bw, y1 + bh, area))

        # Sort by area descending, take top_k
        hotspots.sort(key=lambda roi: roi[4], reverse=True)
        result = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in hotspots[:top_k]]

        # Fallback: if no hotspot found, use center 70% of image
        if not result:
            margin_x = int(w * 0.15)
            margin_y = int(h * 0.15)
            result = [(margin_x, margin_y, w - margin_x, h - margin_y)]
            logger.warning(
                "[DINOv2] No hotspot found, using center fallback ROI: %s",
                result[0],
            )
        else:
            logger.info(
                "[DINOv2] Found %d hotspot(s). Primary ROI: %s",
                len(result),
                result[0],
            )

        return result

    def _preprocess(self, image):
        """Helper to transform image to tensor batch."""
        if isinstance(image, torch.Tensor):
            if image.ndim == 3:
                return image.unsqueeze(0).to(self.device)
            elif image.ndim == 4:
                return image.to(self.device)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Transform PIL image
        return self.transform(image).unsqueeze(0).to(self.device)
