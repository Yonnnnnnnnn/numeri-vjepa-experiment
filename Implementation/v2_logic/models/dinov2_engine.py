"""
DINOv2 Engine (Phase 2)

Wraps Meta's DINOv2 (ViT-B/14) for robust feature extraction and "visual chaos" analysis.
This engine extracts a (768,) semantic vector for density regression and analyzes
feature map variance as a proxy for physical complexity/specularity.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : DINOv2Engine (this module)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <DINOv2Engine>  → __init__ | extract_features | analyze_specularity      │
  │  <Helper>        → _get_transform                                         │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <torch>         ← from torch (tensor ops)                                │
  │  <torch.hub>     ← from torch (model loading)                             │
  │  <transforms>    ← from torchvision (image preprocessing)                 │
  │  <Image>         ← from PIL (image loading)                               │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : "cuda", "cpu", "facebookresearch/dinov2", "dinov2_vitb14"

Production Rules:
  DINOv2Engine    → imports + <DINOv2Engine>
  <DINOv2Engine>  → class DINOv2Engine: <Methods>+
  <Methods>       → __init__(device)
                  | extract_features(image) -> Tensor(768)
                  | analyze_specularity(image) -> float (variance score)
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np


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
