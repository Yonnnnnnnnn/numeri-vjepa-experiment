"""
Test Stateful Event Kernels

Verifies that the EventGeneratorTorch correctly maintains state and generates events
across a sequence of frames.
"""

import torch
import numpy as np
import sys
import os

# Add Implementation root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from v2_logic.kernels.event_gen import EventGeneratorTorch


def test_stateful_events():
    print("Testing Stateful EventGeneratorTorch...")

    H, W = 100, 100
    threshold = 0.2
    gen = EventGeneratorTorch(H, W, threshold=threshold, device="cpu")

    # 1. First frame (Init state)
    frame1 = torch.ones((H, W)) * 0.5
    events1 = gen.process_frame(frame1)
    print(f"Frame 1 (Init): Events sum={events1.sum().item()} (Expected 0)")
    assert events1.sum() == 0

    # 2. Frame 2: Slight increase (Below threshold)
    # log(0.6) - log(0.5) = -0.51 - (-0.69) = 0.18 < 0.2
    frame2 = torch.ones((H, W)) * 0.6
    events2 = gen.process_frame(frame2)
    print(f"Frame 2 (Below Thr): Events sum={events2.sum().item()} (Expected 0)")
    assert events2.sum() == 0

    # 3. Frame 3: Large increase (Above threshold)
    # log(0.8) - log(0.5) = -0.22 - (-0.69) = 0.47 > 0.2
    frame3 = torch.ones((H, W)) * 0.8
    events3 = gen.process_frame(frame3)
    print(f"Frame 3 (Above Thr): Events sum={events3.sum().item()} (Expected H*W)")
    assert events3.sum() == H * W

    # 4. Frame 4: No change
    events4 = gen.process_frame(frame3)
    print(f"Frame 4 (No change): Events sum={events4.sum().item()} (Expected 0)")
    assert events4.sum() == 0

    # 5. Frame 5: Large decrease
    frame5 = torch.ones((H, W)) * 0.3
    # base was updated to log(0.8) approx?
    # v2e statefully updates base_log_frame in increments of threshold.
    # log(0.8) / 0.2 approx.
    events5 = gen.process_frame(frame5)
    print(
        f"Frame 5 (Decrease): Events sum={events5.sum().item()} (Expected negative events)"
    )
    assert torch.all(events5 < 0)

    print("Stateful Event Kernel test PASSED!")


if __name__ == "__main__":
    test_stateful_events()
