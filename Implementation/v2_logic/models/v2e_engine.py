"""
V2E Engine Wrapper

Wraps the SensorsINI/v2e emulator for synthetic event generation.
Pattern: Adapter
- Adapts the v2e EventEmulator to the Glide-and-Count pipeline.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : V2EEngine (this module)

Non-Terminals   :
  ┌─ INTERNAL (defined in this file) ─────────────────────────────────────────┐
  │  <V2EEngine>  → class implementation                                      │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL (imported from other modules) ──────────────────────────────────┐
  │  <EventEmulator> ← from v2ecore.emulator                                  │
  │  <torch>         ← from torch                                             │
  │  <np>            ← from numpy                                             │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : float, int, str, bool, "cuda", "cpu"

Production Rules:
  V2EEngine       → imports + <V2EEngine>
  <V2EEngine>     → __init__ + generate_events + reset
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import numpy as np
import torch
import logging

# Add v2e to path for imports
V2E_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../Techs/v2e-master/v2e-master")
)
if V2E_PATH not in sys.path:
    sys.path.append(V2E_PATH)

try:
    from v2ecore.emulator import EventEmulator
except ImportError:
    # Fallback for different directory structures if needed
    V2E_PATH_ALT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../Techs/v2e-master")
    )
    if V2E_PATH_ALT not in sys.path:
        sys.path.append(V2E_PATH_ALT)
    from v2ecore.emulator import EventEmulator

logger = logging.getLogger(__name__)


class V2EEngine:
    """
    Adapter for the v2e EventEmulator to provide a clean interface
    for synthetic event generation from RGB/Gray frames.

    Pattern: Adapter
    """

    def __init__(
        self,
        pos_thres: float = 0.2,
        neg_thres: float = 0.2,
        sigma_thres: float = 0.03,
        cutoff_hz: float = 0.0,
        leak_rate_hz: float = 0.1,
        shot_noise_rate_hz: float = 0.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the v2e emulator with specified thresholds.
        """
        self.device = device
        self.emulator = EventEmulator(
            pos_thres=pos_thres,
            neg_thres=neg_thres,
            sigma_thres=sigma_thres,
            cutoff_hz=cutoff_hz,
            leak_rate_hz=leak_rate_hz,
            shot_noise_rate_hz=shot_noise_rate_hz,
            device=self.device,
        )
        logger.info("[V2E] Initialized on %s", self.device)

    def generate_events(self, frame: np.ndarray, timestamp: float) -> np.ndarray:
        """
        Synthesize events from a single frame.

        Args:
            frame: np.ndarray (H, W) or (H, W, 3). If color, converts to gray.
            timestamp: float, the time of the frame in seconds.

        Returns:
            np.ndarray: [N, 4] where each row is [timestamp, x, y, polarity]
                       polarity is +1 or -1. Returns empty array if no events.
        """
        # 1. Preprocessing
        if len(frame.shape) == 3:
            # Simple gray conversion if needed, though v2e.py uses cv2.cvtColor(BGR2GRAY)
            # which is faster. Using a simple dot product for robustness here if cv2 not used.
            gray = (
                0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
            )
            gray = gray.astype(np.uint8)
        else:
            gray = frame

        # 2. Emulation
        # generate_events returns torch tensor if device is cuda, or numpy if cpu?
        # Actually in v2ecore.emulator it seems to return numpy in some places or torch.
        # Looking at generate_events implementation again:
        # events = torch.empty((0, 4)...)
        # return events.cpu().numpy() if events.shape[0] > 0 else None

        events = self.emulator.generate_events(gray, timestamp)

        if events is None or len(events) == 0:
            return np.empty((0, 4), dtype=np.float32)

        return events

    def reset(self):
        """Reset the emulator state (e.g. for a new video stream)."""
        self.emulator.reset()
        logger.info("[V2E] State reset")
