"""
V2 Logic Package Init

Ensures environment stability and patches common dependencies.
"""

import sys
import logging

# 🩹 Emergency Patch for Torchvision Circular Import (StochasticDepth)
# This is a known issue in some Torchvision versions when imported in multi-process/HPC environments.
try:
    import torchvision
    import torchvision.ops
    import torch.nn as nn

    if not hasattr(torchvision.ops, "StochasticDepth"):

        class StochasticDepth(nn.Module):
            def __init__(self, p=0.1, mode="batch"):
                super().__init__()

            def forward(self, x):
                return x

        torchvision.ops.StochasticDepth = StochasticDepth
        # We don't print here to avoid cluttering logs during library import
except Exception:
    pass

# Package wide logging configuration
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Export important components for easier access
# CFG Structure:
# ═══════════════════════════════════════════════════════════════════════════════
# Start Symbol    : v2_logic (this package)
# Production Rules:
#   v2_logic  → controllers + models + pipeline + types
# ═══════════════════════════════════════════════════════════════════════════════
