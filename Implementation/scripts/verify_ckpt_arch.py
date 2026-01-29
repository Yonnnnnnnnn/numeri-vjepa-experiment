"""
verify_ckpt_arch.py

Utility script.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : Script (this file)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <Main>           → Script entry point                                    │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <Imports>        ← External dependencies                                 │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : str, int, etc.

Production Rules:
  Script          → Imports + Main
═══════════════════════════════════════════════════════════════════════════════

Pattern: Script
- Standalone executable for utility/testing purposes.

"""
import torch
import os

ckpt_path = "d:/Antigravity/Test VJEPA EVENTBASED LLM/Implementation/checkpoints/vjepa_vitl16.pth.tar"
if os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    predictor_state = checkpoint["predictor"]
    print(f"Predictor state keys: {len(predictor_state)}")
    for k in list(predictor_state.keys()):
        if "predictor_pos_embed" in k:
            print(f"Key: {k}, Shape: {predictor_state[k].shape}")
        if "predictor_blocks" in k and k.endswith("norm1.weight"):
            print(f"Found block key: {k}")
else:
    print("Checkpoint not found")
