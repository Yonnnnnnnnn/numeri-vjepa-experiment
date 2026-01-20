"""
Embedding Engine Module

Uses CLIP to extract visual embeddings for each segmented region.
Enables clustering of similar items.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : EmbeddingEngine (this file)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <EmbeddingEngine>  → CLIP-based embedding extraction                     │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <clip.load>        ← from clip.clip (CLIP model loader)                  │
  │  <clip.tokenize>    ← from clip.clip (text tokenizer)                     │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : numpy.ndarray, torch.Tensor, str

Production Rules:
  EmbeddingEngine → __init__ + embed_regions + embed_text
═══════════════════════════════════════════════════════════════════════════════

Pattern: Facade
- Simplifies CLIP's API for embedding extraction
- Handles preprocessing and batching
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
from typing import List, Optional

# Add clip_src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class EmbeddingEngine:
    """
    DINOv2-based embedding engine for fine-grained visual feature extraction.
    Replaces CLIP to provide better object discrimination.
    """

    def __init__(
        self,
        model_name: str = "vit_large",
        device: str = "cuda",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.transform = None
        self.embed_dim = 1024  # ViT-Large default

        self._load_model(model_name)

    def _load_model(self, model_name: str):
        """Load DINOv2 model from local source."""
        try:
            # Add DINOv2 path - Robust Search Strategy
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # Helper to find Techs folder
            found_dinov2_path = None
            search_dir = current_dir

            print(f"[EmbeddingEngine] Searching for DINOv2 starting from: {search_dir}")

            for i in range(6):  # Search up 6 levels (covers /content/drive/MyDrive/...)
                # Check for standard naming variations
                candidates = [
                    os.path.join(search_dir, "Techs", "dinov2-main", "dinov2-main"),
                    os.path.join(search_dir, "Techs", "dinov2-main"),
                    os.path.join(search_dir, "dinov2-main"),  # If techs isn't used
                ]

                for path in candidates:
                    if os.path.exists(path) and os.path.exists(
                        os.path.join(path, "dinov2")
                    ):
                        found_dinov2_path = path
                        print(
                            f"[EmbeddingEngine] Found candidate path at level {i}: {path}"
                        )
                        break

                if found_dinov2_path:
                    break

                search_dir = os.path.dirname(search_dir)  # Go up one level

            if found_dinov2_path:
                if found_dinov2_path not in sys.path:
                    sys.path.insert(0, found_dinov2_path)

                from dinov2.models.vision_transformer import vit_large

                print(
                    f"[EmbeddingEngine] Loading DINOv2 {model_name} from local source: {found_dinov2_path}..."
                )
                self.model = vit_large(
                    img_size=518, patch_size=14, init_values=1.0, block_chunks=0
                )

            else:
                print(
                    f"[EmbeddingEngine] Warning: Could not find 'Techs/dinov2-main' locally after checking 6 levels up."
                )
                print(
                    "[EmbeddingEngine] Fallback: Attempting to load from torch.hub (requires internet)..."
                )
                # Fallback to torch hub
                self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
                print("[EmbeddingEngine] Loaded from torch.hub success.")
            self.model.to(self.device).eval()

            # Setup transforms
            from torchvision import transforms

            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        (518, 518), interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )

            print(f"[EmbeddingEngine] DINOv2 loaded. Embed dim: {self.embed_dim}")

        except (ImportError, RuntimeError, NameError, AttributeError) as e:
            print(f"[EmbeddingEngine] Error loading DINOv2: {e}")
            import traceback

            traceback.print_exc()
            self.model = None

    def embed_regions(
        self, regions: List[np.ndarray], batch_size: int = 8
    ) -> np.ndarray:
        """
        Extract DINOv2 embeddings for a list of image regions.

        Args:
            regions: List of numpy arrays (H, W, 3) in RGB format
            batch_size: Batch size (smaller for DINOv2 as it's heavy)

        Returns:
            Embeddings array of shape (N, embed_dim)
        """
        if self.model is None or len(regions) == 0:
            return np.zeros((len(regions), self.embed_dim))

        all_embeddings = []

        # Process in batches
        for i in range(0, len(regions), batch_size):
            batch_regions = regions[i : i + batch_size]
            embeddings = self._embed_batch_dinov2(batch_regions)
            all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings, axis=0)

    def _embed_batch_dinov2(self, regions: List[np.ndarray]) -> np.ndarray:
        """Embed batch using DINOv2."""
        images = []
        for region in regions:
            # Convert to PIL
            pil_img = Image.fromarray(region)
            # Apply transforms
            processed = self.transform(pil_img)
            images.append(processed)

        # Stack
        batch = torch.stack(images).to(self.device)

        with torch.no_grad():
            # Forward pass
            # DINOv2 returns dict or tensor depending on impl, usually tensor of CLS token for simple forward
            features = self.model(batch)

            # Normalize
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy()

    def embed_text(self, texts: List[str]) -> np.ndarray:
        """
        Not supported in DINOv2 (Vision Only).
        Returns zeros.
        """
        print("[EmbeddingEngine] Warning: DINOv2 does not support text embedding.")
        return np.zeros((len(texts), self.embed_dim))

    def compute_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity."""
        return np.dot(embedding1, embedding2.T)


if __name__ == "__main__":
    # Quick test
    engine = EmbeddingEngine()
    print("Embedding engine initialized.")
