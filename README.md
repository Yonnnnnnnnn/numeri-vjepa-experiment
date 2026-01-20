# Inference Visualizer: V-JEPA 2 + Event (Zero-Shot)

## Overview

This project implements a **Hybrid Event-Based Inference Visualizer**. It converts standard video into simulated Event streams (using a high-performance Triton Kernel) and feeds them alongside RGB keyframes into a **V-JEPA 2 + LLM** model to perform counting and reasoning tasks (e.g., inventory counting).

**Strategy:** Zero-Shot Inference (No Training Required).
**Hardware Target:** NVIDIA L40S (RunPod).

## Project Structure

```
.
├── src/
│   ├── kernels/     # Event Generation Logic (Triton/PyTorch)
│   ├── models/      # Hybrid V-JEPA Model Wrapper
│   └── pipeline/    # Main Inference Loop & Visualization
├── main.py          # CLI Entry Point
├── requirements.txt # Python Dependencies
└── README.md        # This file
```

## Deployment Guide (RunPod)

### 1. Rent a GPU

- Go to [RunPod.io](https://runpod.io).
- Rent a **Secure Cloud** instance with **NVIDIA L40S** (or L4/A100).
- Image: Use standard `PyTorch 2.1` or `2.2` template.

### 2. Setup (On RunPod Terminal)

```bash
# 1. Clone/Upload this specific folder to /workspace/

# 2. Run the Setup Script (Handles System Deps + Pip + Models)
bash setup.sh
```

### 3. Model Setup (Important)

The system uses **Vicuna-7B** (LLM) and **CLIP-Large** (Vision). These will download automatically on the first run, but it is recommended to pre-download them, especially for Docker builds.

```bash
# Pre-download weights to ~/.cache/huggingface
python scripts/download_models.py
```

**Note on V-JEPA:**
This codebase currently uses **CLIP as a proxy** for V-JEPA 2 because V-JEPA's weights are not yet standard in the Transformers library. In a production scenario with private access, you would plug `vjepa2_huge.pth` into `src/models/hybrid_vjepa.py`.

### 4. Running the Visualizer

Upload a test video (e.g., `inventory_test.mp4`) to the folder.

```bash
# Basic Run (Zero-Shot)
python main.py --video inventory_test.mp4 --threshold 0.1 --output result.mp4
```

**Arguments:**

- `--video`: Path to input video.
- `--threshold`: Sensitivity for Event Generation (Lower = More partial events, Higher = Cleaner but less info). Default `0.1`.
- `--output`: Filename for the side-by-side visualization.

## Troublehooting

- **Triton Error:** If Triton fails, the code automatically falls back to PyTorch (Slower but works).
- **Model Load Error:** This prototype uses `vicuna-7b` and `clip-vit` as proxies. Ensure you have HuggingFace access token set if switching to gated models (`huggingface-cli login`).

## Roadmap / Next Steps

- [ ] Test with Real Inventory Footage.
- [ ] Integrate `SensorsINI/v2e` for higher fidelity simulation if needed.
- [ ] Fine-tune Adapter if Zero-Shot accuracy is low.
