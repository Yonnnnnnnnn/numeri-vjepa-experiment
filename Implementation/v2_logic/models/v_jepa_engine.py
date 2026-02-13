"""
V-JEPA World Model Engine (Brain)

Interfaces with the Meta FAIR V-JEPA codebase for 3D spatial understanding
and object permanence. Includes PersistentLatentContext for temporal memory.
Pattern: Adapter
- Adapts the V-JEPA ViT and Predictor for the Glide-and-Count pipeline.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : VJEPAEngine (this module)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <PersistentLatentContext> → Sliding-window frame buffer (T=16)          │
  │  <VJEPAEngine>             → Adapter over ViT + Predictor               │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <VisionTransformer> ← from vjepa_src.models.vision_transformer          │
  │  <Predictor>         ← from vjepa_src.models.predictor                   │
  │  <deque>             ← from collections (circular buffer)                │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : str, float, int, bool, "cuda", "cpu", 16, 224, 1024

Production Rules:
  VJEPAEngine              → imports + <PersistentLatentContext> + <VJEPAEngine>
  <PersistentLatentContext> → __init__ + update + get_context_tensor + reset + is_ready
  <VJEPAEngine>            → __init__ + encode + encode_context + extract_object_latent
                              + predict_trajectory + load_weights + reset
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import os
from collections import deque

import torch
import torch.nn.functional as F

# Handle V-JEPA imports from refactored package
# Pattern: Adapter - ensuring internal Meta imports work without shadowing
logger = logging.getLogger(__name__)

try:
    from vjepa_src.models.vision_transformer import vit_large
    from vjepa_src.models.predictor import vit_predictor
except ImportError:
    # Fallback to direct imports if path is already set (primarily for local dev)
    try:
        from models.vision_transformer import vit_large
        from models.predictor import vit_predictor
    except ImportError:
        logger.error("[V-JEPA] Failed to find vjepa_src or models package.")
        raise


class PersistentLatentContext:
    """
    Sliding-window frame buffer that accumulates raw frames over time,
    providing a temporal context tensor for V-JEPA encoding.

    Pattern: Observer (accumulates state passively from the video stream)

    The buffer holds up to QUEUE_SIZE frames. When full, the oldest frame
    is discarded. The context tensor is always shaped (1, C, T, H, W)
    where T=QUEUE_SIZE, ready for V-JEPA's ViT encoder.

    Attributes:
        queue_size: Number of frames to keep in the sliding window.
        buffer: Circular buffer of frame tensors.
    """

    QUEUE_SIZE = 16  # Matched to V-JEPA ViT-Large num_frames

    def __init__(self, queue_size: int = 16):
        self.queue_size = queue_size
        self.buffer: deque = deque(maxlen=queue_size)

    def update(self, frame: torch.Tensor) -> None:
        """
        Add a new frame to the sliding window.

        Args:
            frame: (C, H, W) or (1, C, H, W) normalized [0,1] tensor.
        """
        if frame.ndim == 4:
            frame = frame.squeeze(0)
        if frame.ndim != 3:
            raise ValueError(
                f"[PersistentLatentContext] Expected 3D tensor (C, H, W), "
                f"got shape {frame.shape}"
            )
        self.buffer.append(frame)

    def get_context_tensor(self) -> torch.Tensor:
        """
        Build the (1, C, T, H, W) context tensor from the buffer.

        If the buffer has fewer than queue_size frames, the last frame
        is repeated to fill the temporal dimension.

        Returns:
            Tensor of shape (1, C, T, H, W).
        """
        if len(self.buffer) == 0:
            raise RuntimeError(
                "[PersistentLatentContext] Buffer is empty. "
                "Call update() with at least one frame first."
            )

        frames = list(self.buffer)

        # Pad with last frame if not enough frames yet
        while len(frames) < self.queue_size:
            frames.append(frames[-1])

        # Stack: List[(C, H, W)] -> (C, T, H, W) -> (1, C, T, H, W)
        context = torch.stack(frames, dim=1).unsqueeze(0)
        return context

    @property
    def is_ready(self) -> bool:
        """True if the buffer has at least one frame."""
        return len(self.buffer) > 0

    @property
    def is_full(self) -> bool:
        """True if the buffer has reached queue_size frames."""
        return len(self.buffer) == self.queue_size

    @property
    def frame_count(self) -> int:
        """Number of frames currently in the buffer."""
        return len(self.buffer)

    def reset(self) -> None:
        """Clear the frame buffer."""
        self.buffer.clear()


class VJEPAEngine:
    """
    World Model engine for 3D spatial reasoning and object permanence.

    Pattern: Adapter
    - Wraps Meta V-JEPA ViT-Large encoder and predictor.
    - Provides PersistentLatentContext for temporal memory across frames.
    - Exposes spatial latent extraction for ReID and Golden Alpha matching.
    """

    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"

        # 1. Initialize Encoder (ViT-Large is standard for V-JEPA)
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

        # 3. Persistent Context Memory (V3.1 upgrade)
        self.context = PersistentLatentContext(queue_size=16)
        self.latent_context = None  # Most recent latent map from encoder

        logger.info("[V-JEPA] Initialized on %s", self.device)

        # 4. Auto-load weights if available
        default_ckpt = os.path.join(
            os.path.dirname(__file__), "../../checkpoints/vjepa_vitl16.pth.tar"
        )
        if os.path.exists(default_ckpt):
            self.load_weights(default_ckpt)

    def encode(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode raw frame/event tensors into JEPA latent space.
        Also updates the PersistentLatentContext buffer.

        Args:
            frame_tensor: (B, C, H, W) normalized [0,1]

        Returns:
            Latent tensor from the ViT encoder.
        """
        with torch.no_grad():
            # Update persistent context with the new frame
            if frame_tensor.ndim == 4 and frame_tensor.shape[0] == 1:
                self.context.update(frame_tensor.squeeze(0).cpu())

            # Build temporal context from sliding window
            context_tensor = self.context.get_context_tensor()
            context_tensor = context_tensor.to(self.device)

            latent = self.encoder(context_tensor)
            self.latent_context = latent
        return latent

    def encode_context(self) -> torch.Tensor:
        """
        Encode the full PersistentLatentContext buffer without adding new frames.
        Useful for re-encoding after parameter changes.

        Returns:
            Latent tensor from the ViT encoder.
        """
        if not self.context.is_ready:
            raise RuntimeError("[V-JEPA] Cannot encode_context: no frames in buffer.")

        with torch.no_grad():
            context_tensor = self.context.get_context_tensor().to(self.device)
            latent = self.encoder(context_tensor)
            self.latent_context = latent
        return latent

    def extract_object_latent(
        self, bbox: dict, latent_map: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Extract latent features for a specific object region (Latent-Pose).

        Crops the spatial latent map (B, N_patches, D) to the bbox region,
        then averages the patch embeddings to get a single feature vector.

        This is the key method for ReID: it gives each object a "latent pose"
        that can be compared across frames for identity matching.

        Args:
            bbox: {'x': float, 'y': float, 'w': float, 'h': float}
                  Coordinates normalized to [0, 1] relative to the image.
            latent_map: Optional override. If None, uses self.latent_context.

        Returns:
            (D,) feature vector for the object.
        """
        if latent_map is None:
            latent_map = self.latent_context

        if latent_map is None:
            # Try to recover from disk (Process-Sharing fallback)
            ctx_path = "vjepa_context_dump.pt"
            if os.path.exists(ctx_path):
                try:
                    loaded = torch.load(ctx_path, map_location=self.device)
                    self.latent_context = loaded
                    latent_map = loaded
                    logging.info("[V-JEPA] Recovered latent context from %s", ctx_path)
                except Exception as e:
                    logging.warning("[V-JEPA] Failed to recover context: %s", e)

        if latent_map is None:
            raise RuntimeError(
                "[V-JEPA] No latent context available. Call encode() first or ensure context dump exists."
            )

        with torch.no_grad():
            # V-JEPA ViT outputs (B, N_patches, D) where N_patches = T' * H' * W'
            # For ViT-L with 224px, 16px patch: H' = W' = 14
            # For video: T' = num_frames / tubelet_size = 16 / 2 = 8
            spatial_h = 14
            spatial_w = 14
            temporal_t = 8
            embed_dim = latent_map.shape[-1]

            # Reshape to (B, T', H', W', D)
            latent_5d = latent_map.view(
                latent_map.shape[0], temporal_t, spatial_h, spatial_w, embed_dim
            )

            # Average over temporal dimension -> (B, H', W', D)
            latent_spatial = latent_5d.mean(dim=1)

            # Convert normalized bbox to patch coordinates
            patch_x = int(bbox["x"] * spatial_w)
            patch_y = int(bbox["y"] * spatial_h)
            patch_w = max(1, int(bbox["w"] * spatial_w))
            patch_h = max(1, int(bbox["h"] * spatial_h))

            # Clamp to valid range
            patch_x = max(0, min(patch_x, spatial_w - 1))
            patch_y = max(0, min(patch_y, spatial_h - 1))
            patch_x_end = min(patch_x + patch_w, spatial_w)
            patch_y_end = min(patch_y + patch_h, spatial_h)

            # Crop and average -> (D,)
            region = latent_spatial[0, patch_y:patch_y_end, patch_x:patch_x_end, :]
            object_latent = region.mean(dim=(0, 1))

            # L2 normalize for cosine similarity downstream
            object_latent = F.normalize(object_latent, dim=0)

        return object_latent

    def predict_trajectory(self, steps: int = 1) -> torch.Tensor:
        """
        Predict future latent states to handle occlusions (Permanence).

        Uses the V-JEPA predictor to anticipate where objects will be
        in latent space, even when they are temporarily hidden.

        Args:
            steps: Number of future steps to predict (reserved for future use).

        Returns:
            Predicted latent tensor, or None if no context exists.
        """
        _ = steps
        if self.latent_context is None:
            return None

        with torch.no_grad():
            prediction = self.predictor(
                self.latent_context,
                self.latent_context,
                masks_ctxt=None,
                masks_tgt=None,
            )
        return prediction

    def load_weights(self, checkpoint_path: str) -> None:
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
            "[V-JEPA] Successfully loaded and aligned weights from %s",
            checkpoint_path,
        )

    def reset(self) -> None:
        """Reset both the latent context and the persistent frame buffer."""
        self.latent_context = None
        self.context.reset()
        logger.info("[V-JEPA] Context and buffer reset")

    def export_context(self, path: str = "vjepa_context_dump.pt") -> None:
        """Export current latent context to disk for external visualizers."""
        if self.latent_context is not None:
            try:
                torch.save(self.latent_context, path)
            except Exception as e:
                logger.warning("[V-JEPA] Failed to export context: %s", e)
