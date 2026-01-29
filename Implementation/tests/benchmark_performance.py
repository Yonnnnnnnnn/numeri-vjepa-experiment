"""
Performance Benchmark Script

Measures execution time of key components.
Target: Volume Calc < 50ms, Loop Latency < 100ms.
"""

import time
import numpy as np
import pytest
from v2_logic.utils.math_utils import MathUtils


def benchmark_volume_heuristic():
    """Measure MathUtils.estimate_volume_heuristic speed."""
    print("\n[Benchmark] estimate_volume_heuristic")

    # 640x480 Depth Map
    depth_map = np.random.rand(480, 640).astype(np.float32)
    mask = np.ones((480, 640), dtype=np.uint8)  # Full mask worst case?
    # Make mask smaller for realistic case (e.g., 20% coverage)
    mask[0:200, 0:200] = 0

    start_time = time.time()
    for _ in range(100):
        MathUtils.estimate_volume_heuristic(depth_map, mask)
    end_time = time.time()

    avg_time = (end_time - start_time) / 100
    print(f"  Avg Time (100 runs): {avg_time*1000:.2f} ms")

    if avg_time > 0.05:
        print("  WARNING: Exceeds 50ms target!")
    else:
        print("  PASS: Within limits.")


if __name__ == "__main__":
    benchmark_volume_heuristic()
