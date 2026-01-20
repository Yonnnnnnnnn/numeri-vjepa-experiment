#!/bin/bash

# Stop on error
set -e

echo "========================================"
echo "  V-JEPA Visualizer: Setup Script"
echo "========================================"

# 1. System Dependencies (FFmpeg is critical for VideoReader/OpenCV)
echo "[1/3] Updating System Packages..."
apt-get update -y
apt-get install -y ffmpeg libsm6 libxext6 git wget

# 2. Python Dependencies
echo "[2/3] Installing Python Requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Download Models (Optional but recommended)
echo "[3/3] Pre-downloading Models..."
if [ -f "scripts/download_models.py" ]; then
    python scripts/download_models.py
else
    echo "Warning: scripts/download_models.py not found. Skipping download."
fi

echo "========================================"
echo "  Setup Complete! Ready to Run."
echo "========================================"
