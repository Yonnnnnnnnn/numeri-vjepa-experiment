import torch
import os

ckpt_path = "d:/Antigravity/Test VJEPA EVENTBASED LLM/Implementation/checkpoints/vjepa_vitl16.pth.tar"
if not os.path.exists(ckpt_path):
    print(f"Checkpoint not found: {ckpt_path}")
else:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    print(f"Type of checkpoint: {type(checkpoint)}")
    if isinstance(checkpoint, dict):
        print(f"Keys in checkpoint: {checkpoint.keys()}")
        if "encoder" in checkpoint:
            print(f"Sample encoder keys: {list(checkpoint['encoder'].keys())[:10]}")
        if "predictor" in checkpoint:
            print(f"Sample predictor keys: {list(checkpoint['predictor'].keys())[:10]}")
        if "model" in checkpoint:
            print(f"Sample model keys: {list(checkpoint['model'].keys())[:10]}")

        # If it's a flat dict
        print(f"Sample flat keys: {list(checkpoint.keys())[:10]}")
