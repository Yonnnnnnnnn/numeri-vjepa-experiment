"""
Event Generation Kernels
Implements converting RGB Frames to Event representations (Log-Diff)
using both Pure PyTorch and OpenAI Triton for L40S acceleration.
"""

import torch
import math

# Try importing Triton, handle case if not available (e.g. Windows dev env)
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Warning: Triton not found. Using PyTorch fallback.")

# ==========================================
# 1. PyTorch Implementation (Stateful DVS)
# ==========================================


class EventGeneratorTorch:
    """
    Stateful Event Generator using PyTorch.
    Mimes the v2e accumulator logic.
    """

    def __init__(self, height, width, threshold=0.2, device="cuda"):
        self.height = height
        self.width = width
        self.threshold = threshold
        self.device = device
        # Memorized log-intensity per pixel
        self.base_log_frame = None
        self.epsilon = 1e-6

    def reset(self):
        self.base_log_frame = None

    def process_frame(self, frame: torch.Tensor):
        """
        Args:
            frame: (H, W) or (C, H, W) normalized [0,1]
        Returns:
            events: (H, W) with values {-1, 0, 1}
        """
        if len(frame.shape) == 3:
            # RGB to Gray
            gray = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
        else:
            gray = frame

        log_frame = torch.log(gray + self.epsilon)

        if self.base_log_frame is None:
            self.base_log_frame = log_frame.clone()
            return torch.zeros_like(log_frame)

        # Difference from memorized state
        diff = log_frame - self.base_log_frame

        events = torch.zeros_like(diff)

        # ON Events
        on_mask = diff > self.threshold
        events[on_mask] = 1.0
        # Update base_log_frame only where events occurred
        # In a real DVS, it resets by 'threshold' units
        # Here we simplify: if multiple thresholds crossed, we count as one in this frame
        # OR we could loop. v2e loops until diff < threshold.

        # Simplified "Iterative" update
        while torch.any(diff > self.threshold):
            mask = diff > self.threshold
            self.base_log_frame[mask] += self.threshold
            diff[mask] -= self.threshold
            # Note: For simplicity in a single-frame kernel, we just mark polarity 1.0
            # A more complex kernel would return a list of spikes per pixel.
            events[mask] = 1.0

        while torch.any(diff < -self.threshold):
            mask = diff < -self.threshold
            self.base_log_frame[mask] -= self.threshold
            diff[mask] += self.threshold
            events[mask] = -1.0

        return events


def rgb_to_event_torch(
    current_frame: torch.Tensor, prev_frame: torch.Tensor, threshold: float = 0.1
):
    """
    Legacy stateless wrapper for backward compatibility.
    """
    epsilon = 1e-6
    diff = torch.log(current_frame + epsilon) - torch.log(prev_frame + epsilon)
    events = torch.zeros_like(diff)
    events[diff > threshold] = 1.0
    events[diff < -threshold] = -1.0
    return events


# ==========================================
# 2. Triton Kernel Implementation (L40S Optimized)
# ==========================================

if HAS_TRITON:

    @triton.jit
    def log_diff_kernel(
        x_ptr,
        y_ptr,
        out_ptr,
        n_elements,
        threshold,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Triton Kernel for Element-wise Log Difference (Stateless).
        """
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)

        epsilon = 1e-6
        diff = tl.log(x + epsilon) - tl.log(y + epsilon)

        output = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        output = tl.where(diff > threshold, 1.0, output)
        output = tl.where(diff < -threshold, -1.0, output)

        tl.store(out_ptr + offsets, output, mask=mask)

    def rgb_to_event_triton(
        current_frame: torch.Tensor, prev_frame: torch.Tensor, threshold: float = 0.1
    ):
        """
        Wrapper to call Triton kernel.
        """
        assert current_frame.is_cuda and prev_frame.is_cuda
        n_elements = current_frame.numel()
        output = torch.empty_like(current_frame)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        log_diff_kernel[grid](
            current_frame, prev_frame, output, n_elements, threshold, BLOCK_SIZE=1024
        )
        return output

else:

    def rgb_to_event_triton(current_frame, prev_frame, threshold=0.1):
        return rgb_to_event_torch(current_frame, prev_frame, threshold)
