"""
test_dinov2.py

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
import sys
import os
import torch

# Add Techs directory to path to find dinov2-main
techs_path = os.path.abspath(
    os.path.join(os.getcwd(), "..", "Techs", "dinov2-main", "dinov2-main")
)
sys.path.insert(0, techs_path)

print(f"Added to path: {techs_path}")

try:
    from dinov2.models.vision_transformer import vit_large

    print("Successfully imported vit_large from dinov2")

    model = vit_large(img_size=518, patch_size=14, init_values=1.0, block_chunks=0)
    print("Successfully instantiated model")

    # Test forward pass with dummy data
    dummy_input = torch.randn(1, 3, 518, 518)
    output = model(dummy_input)
    print(f"Forward pass successful. Output shape: {output.shape}")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
