"""
Main Entry Point for Inference Visualizer

CLI for running video-to-event inference using Hybrid V-JEPA.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : MainVisualizer (this file)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <MainFunction>   → main CLI entry point                                  │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <argparse>       ← from stdlib (Argument Parsing)                        │
  │  <run_v2_visualizer> ← from v2_logic.pipeline.engine_v2                   │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : str, float, args

Production Rules:
  MainVisualizer → imports + <MainFunction>
  <MainFunction> → parse_args + run_v2_visualizer
═══════════════════════════════════════════════════════════════════════════════

Pattern: Command (Script)
- Acts as the CLI entry point invoking the pipeline command.
"""

import argparse
import os
import sys

# 🩹 Emergency Patch for Torchvision Circular Import (StochasticDepth)
# This MUST happen before any torchvision/torch imports
try:
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.ops

    if not hasattr(torchvision.ops, "StochasticDepth"):

        class StochasticDepth(nn.Module):
            def __init__(self, p=0.1, mode="batch"):
                super().__init__()

            def forward(self, x):
                return x

        torchvision.ops.StochasticDepth = StochasticDepth
except Exception:
    pass

import logging

# Configure logging to show info messages in Colab/Console
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

# Add Implementation root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Robustly add Techs dependencies (required for Colab subprocesses)
TECHS_PATHS = [
    os.path.join(PARENT_DIR, "Techs/Depth-Anything-V2-main/Depth-Anything-V2-main"),
    os.path.join(PARENT_DIR, "Techs/CountVid-main/CountVid-main"),
    os.path.join(PARENT_DIR, "Techs/sam2-main/sam2-main"),
    os.path.join(PARENT_DIR, "Techs/v2e-master/v2e-master"),
]

for p in TECHS_PATHS:
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)

from v2_logic.pipeline.engine_v2 import run_v2_visualizer


def main():
    parser = argparse.ArgumentParser(
        description="Run V2 Glide-and-Count Inference Visualizer"
    )
    parser.add_argument(
        "--video", type=str, required=True, help="Path to input video file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_v2_visualizer.mp4",
        help="Path to output visualization file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Log-diff threshold for event generation (Sensitivity)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="User prompt for brand-specific SKU discovery (e.g. 'count Baked beans, Amoy, Custard')",
    )

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found at {args.video}")
        sys.exit(1)

    print("=" * 50)
    print(" INFERENCE VISUALIZER: GLIDE-AND-COUNT")
    print("=" * 50)
    print(f" Source Video   : {args.video}")
    print(f" Output target  : {args.output}")
    print(f" Sensitivity    : {args.threshold}")
    if args.prompt:
        print(f" Prompt         : {args.prompt}")
    print("-" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Execution Unit : {device.upper()}")
    if device == "cuda":
        print(f" GPU Model      : {torch.cuda.get_device_name(0)}")
        print(
            f" VRAM Available : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        print(" ⚠️ WARNING: Running on CPU. Performance will be very slow.")
    print("-" * 50)

    try:
        run_v2_visualizer(
            args.video,
            args.output,
            args.threshold,
            device=device,
            prompt=args.prompt,
        )
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        print(f"\nINFERENCE ERROR: {e}")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"\nCRITICAL ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
