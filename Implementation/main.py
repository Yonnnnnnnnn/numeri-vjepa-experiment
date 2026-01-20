"""
Main Entry Point for Inference Visualizer
CLI for running video-to-event inference using Hybrid V-JEPA.
"""

import argparse
import os
import sys

import torch
import logging

# Configure logging to show info messages in Colab/Console
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

# Add local directory to path to ensure modules are found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
        run_v2_visualizer(args.video, args.output, args.threshold, device=device)
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
