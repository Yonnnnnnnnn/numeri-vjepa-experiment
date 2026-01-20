"""
Pre-download Models for RunPod/Colab
Run this script during build/setup to cache models.
Downloads: VLM (Qwen2.5-VL), SAM2 checkpoint, CLIP
"""

import os
import torch
import urllib.request
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# Model Configuration
# ══════════════════════════════════════════════════════════════════════════════

VLM_ID = "Qwen/Qwen2-VL-7B-Instruct"
CLIP_ID = "openai/clip-vit-base-patch32"

SAM2_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
)
SAM2_CHECKPOINT_NAME = "sam2.1_hiera_tiny.pt"


def get_checkpoint_dir():
    """Get the directory for SAM2 checkpoints."""
    # Try relative to script, then fall back to home
    script_dir = Path(__file__).parent.parent
    ckpt_dir = script_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    return ckpt_dir


def download_sam2_checkpoint():
    """Download SAM2 checkpoint if not exists."""
    ckpt_dir = get_checkpoint_dir()
    ckpt_path = ckpt_dir / SAM2_CHECKPOINT_NAME

    if ckpt_path.exists():
        print(f"      Already exists: {ckpt_path}")
        return str(ckpt_path)

    print(f"      Downloading from {SAM2_CHECKPOINT_URL}...")
    print(f"      Saving to {ckpt_path}...")

    try:
        urllib.request.urlretrieve(SAM2_CHECKPOINT_URL, str(ckpt_path))
        print(f"      Success ✓ ({ckpt_path.stat().st_size / 1e6:.1f} MB)")
        return str(ckpt_path)
    except Exception as e:
        print(f"      Failed ✗: {e}")
        return None


def download_vlm():
    """Download Qwen2.5-VL model."""
    try:
        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

        Qwen2VLProcessor.from_pretrained(VLM_ID)
        Qwen2VLForConditionalGeneration.from_pretrained(
            VLM_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        print("      Success ✓")
    except Exception as e:
        print(f"      Failed ✗: {e}")


def download_dinov2():
    """Download DINOv2 model via torch.hub."""
    try:
        print("      Loading DINOv2 from torch.hub (facebookresearch/dinov2)...")
        # This caches the model in ~/.cache/torch/hub/checkpoints
        torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        print("      Success ✓")
    except Exception as e:
        print(f"      Failed ✗: {e}")


def download_models():
    """Download all required models."""
    print("=" * 60)
    print(" MODEL DOWNLOADER")
    print("=" * 60)

    # 1. SAM2 Checkpoint (~40MB)
    print(f"\n[1/3] Downloading SAM2 Checkpoint: {SAM2_CHECKPOINT_NAME}")
    sam_path = download_sam2_checkpoint()

    # 2. DINOv2 (~1GB)
    print(f"\n[2/3] Downloading DINOv2 (ViT-Large)...")
    download_dinov2()

    # 3. VLM (~14GB)
    print(f"\n[3/3] Downloading VLM: {VLM_ID}")
    print("      (This may take 10-15 minutes for ~14GB model...)")
    download_vlm()

    print("\n" + "=" * 60)
    print(" DOWNLOAD COMPLETE")
    print("=" * 60)

    if sam_path:
        print(f"\nSAM2 checkpoint saved at: {sam_path}")
        print(f'Use with: --sam-checkpoint "{sam_path}"')


if __name__ == "__main__":
    download_models()
