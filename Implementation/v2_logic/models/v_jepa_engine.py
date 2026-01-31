"""
V-JEPA World Model Engine (Brain)

Interfaces with the Meta FAIR V-JEPA codebase for 3D spatial understanding
and object permanence.
Pattern: Adapter
- Adapts the V-JEPA ViT and Predictor for the Glide-and-Count pipeline.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : VJEPAEngine (this module)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <VJEPAEngine> → class implementation                                     │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <VisionTransformer> ← from vjepa_src.models.vision_transformer           │
  │  <Predictor>         ← from vjepa_src.models.predictor                      │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : str, float, int, bool, "cuda", "cpu"

Production Rules:
  VJEPAEngine     → imports + <VJEPAEngine>
  <VJEPAEngine>   → __init__ + encode + predict_next_state + reset
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import os

import torch

# Handle V-JEPA imports from refactored package
# Pattern: Adapter - ensuring internal Meta imports work without shadowing
logger = logging.getLogger(__name__)

try:
    from vjepa_src.models.vision_transformer import vit_large
    from vjepa_src.models.predictor import vit_predictor

    # from v2_logic.models.temporal_memory import TemporalContext # pylint: disable=unused-import
except ImportError:
    # Fallback to direct imports if path is already set (primarily for local dev)
    try:
        from models.vision_transformer import vit_large
        from models.predictor import vit_predictor
    except ImportError:
        logger.error("[V-JEPA] Failed to find vjepa_src or models package.")
        raise


class VJEPAEngine:
    """
    World Model engine for 3D spatial reasoning and object permanence.

    Pattern: Adapter
    """

    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"

        # 1. Initialize Encoder (ViT-Large is standard for V-JEPA)
        # Using vit_large from jepa-main
        self.encoder = (
            vit_large(
                img_size=224,
                patch_size=16,
                num_frames=16,  # Matched to 5.1GB ViT-L checkpoint (1568 patches)
            )
            .to(self.device)
            .eval()
        )

        # 2. Initialize Predictor
        self.predictor = (
            vit_predictor(
                embed_dim=1024,  # ViT-Large dim
                predictor_embed_dim=384,  # Matched to 5.1GB checkpoint
                num_frames=16,  # Matched to 1568 patches (1568 = 8 * 14*14)
                tubelet_size=2,  # Temporal downsampling
                depth=12,  # Matched to checkpoint
                num_heads=16,
                use_mask_tokens=True,  # Matched to checkpoint
            )
            .to(self.device)
            .eval()
        )
        # 3. Context Memory
        self.latent_context = None

        logger.info("[V-JEPA] Initialized on %s", self.device)

        # 4. Auto-load weights if available
        default_ckpt = os.path.join(
            os.path.dirname(__file__), "../../checkpoints/vjepa_vitl16.pth.tar"
        )
        if os.path.exists(default_ckpt):
            self.load_weights(default_ckpt)

    def encode(self, frame_tensor: torch.Tensor):
        """
        Encode raw frame/event tensors into JEPA latent space.

        Args:
            frame_tensor: (B, C, H, W) normalized [0,1]
        """
        with torch.no_grad():
            # V-JEPA (vit_large for video) expects 5D: (B, C, T, H, W)
            # We must provide at least 'tubelet_size' frames, ideally matched to num_frames (16).
            # If we only have 1 frame, we repeat it to avoid interpolating to D=0.
            if frame_tensor.ndim == 4:
                # Expects (B, C, T, H, W)
                frame_tensor = frame_tensor.unsqueeze(2).repeat(1, 1, 16, 1, 1)
            elif frame_tensor.ndim == 5 and frame_tensor.shape[2] == 1:
                frame_tensor = frame_tensor.repeat(1, 1, 16, 1, 1)

            latent = self.encoder(frame_tensor.to(self.device))
            self.latent_context = latent
        return latent

    def predict_next_state(self, steps=1):
        """
        Predict future latent states to handle occlusions (Permanence).

        Args:
            steps: Number of future steps to predict.
        """
        _ = steps  # pylint: disable=unused-argument
        if self.latent_context is None:
            return None

        with torch.no_grad():
            # In V-JEPA, the predictor takes context and target masks.
            # For permanence, we interpret the "future" as masked regions.
            # This is a simplification of the official vjepa training loop for inference.
            prediction = self.predictor(
                self.latent_context,
                self.latent_context,  # Predictor uses context to reconstruct tgt
                masks_ctxt=None,  # In inference, we might provide full context
                masks_tgt=None,
            )
        return prediction

    def load_weights(self, checkpoint_path: str):
        """Load pretrained V-JEPA weights."""
        if not os.path.exists(checkpoint_path):
            logger.warning("[V-JEPA] Checkpoint not found: %s", checkpoint_path)
            return

        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Handle prefixes from DistributedDataParallel and repo-specific wrapping
        def fix_state_dict(state_dict):
            new_dict = {}
            for k, v in state_dict.items():
                name = k
                if name.startswith("module."):
                    name = name[7:]
                if name.startswith("backbone."):
                    name = name[9:]
                new_dict[name] = v
            return new_dict

        encoder_state = fix_state_dict(checkpoint["encoder"])
        self.encoder.load_state_dict(encoder_state, strict=True)

        predictor_state = fix_state_dict(checkpoint["predictor"])
        self.predictor.load_state_dict(predictor_state, strict=True)

        logger.info(
            "[V-JEPA] Successfully loaded and aligned weights from %s", checkpoint_path
        )

    def reset(self):
        self.latent_context = None
        logger.info("[V-JEPA] Context reset")
