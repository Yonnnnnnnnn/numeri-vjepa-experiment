"""
Fusion Engine V2 (Spike-Mask Residue Detection)

Fuses V2E spike data with SAM2 masks to detect anomalies.
Core component of the Recursive Intent "Triple Check" system.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : FusionEngineV2 (this module)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <FusionEngineV2>  → Main fusion logic for anomaly detection              │
  │  <FusionResult>    → Output dataclass with residue and anomaly info       │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <PerceptionState>  ← from types.graph_state (State container)            │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : numpy.ndarray, float, int, List, Dict

Production Rules:
  FusionEngineV2     → __init__ + fuse_spike_mask + detect_anomaly
  fuse_spike_mask    → subtract_masks + calculate_residue → FusionResult
  detect_anomaly     → check_spatial + check_volumetric → anomaly_type
═══════════════════════════════════════════════════════════════════════════════

Pattern: Strategy
- Encapsulates the spike-mask fusion algorithm.
- Can be extended for different fusion strategies.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    """Result of spike-mask fusion analysis."""

    # Spike-Mask Fusion
    residual_spike_energy: float  # Energy in unexplained regions
    total_spike_energy: float  # Total spike energy in frame
    residue_ratio: float  # residual / total (0-1)

    # Unexplained Regions
    unexplained_blobs: List[Dict[str, Any]]  # List of unexplained regions

    # Anomaly Detection
    has_spatial_anomaly: bool  # High residual spike (something moving outside masks)
    has_volumetric_anomaly: bool  # Count doesn't match volume range

    # Motion Compensation (for filtering camera jitter)
    motion_compensated: bool
    camera_motion_energy: float


@dataclass
class MotionState:
    """Tracks motion history for camera jitter detection."""

    prev_frame_mean: Optional[float] = None
    motion_history: List[float] = field(default_factory=list)
    max_history: int = 10


class FusionEngineV2:
    """
    Spike-Mask Fusion Engine for anomaly detection.

    Implements:
    1. Mask Subtraction: spike_map - (spike_map * combined_mask)
    2. Residue Calculation: Energy in unexplained regions
    3. Motion Compensation: Filter out camera jitter

    Pattern: Strategy
    """

    def __init__(
        self,
        residue_threshold: float = 0.15,  # 15% residue triggers anomaly
        motion_threshold: float = 0.05,  # 5% global motion = camera jitter
        min_blob_area: int = 100,  # Minimum blob area to consider
    ):
        """
        Args:
            residue_threshold: Ratio threshold for spatial anomaly.
            motion_threshold: Global motion threshold for camera jitter.
            min_blob_area: Minimum area for unexplained blob detection.
        """
        self.residue_threshold = residue_threshold
        self.motion_threshold = motion_threshold
        self.min_blob_area = min_blob_area
        self.motion_state = MotionState()

    def fuse_spike_mask(
        self,
        spike_map: np.ndarray,
        masks: List[np.ndarray],
        n_visible: int,
        n_volumetric_range: Tuple[int, int],
    ) -> FusionResult:
        """
        Fuse spike map with segmentation masks.

        Args:
            spike_map: V2E spike energy map (H, W), values 0-1.
            masks: List of binary masks from SAM2.
            n_visible: Visual count from CountGD.
            n_volumetric_range: (min, max) count from volumetric estimation.

        Returns:
            FusionResult with residue analysis and anomaly detection.
        """
        h, w = spike_map.shape[:2] if len(spike_map.shape) >= 2 else (480, 640)

        # Calculate total spike energy
        total_energy = float(np.sum(spike_map))

        # Create combined mask from all segmentation masks
        if masks and len(masks) > 0:
            combined_mask = np.zeros((h, w), dtype=np.float32)
            for mask in masks:
                if mask.shape == combined_mask.shape:
                    combined_mask = np.maximum(combined_mask, mask.astype(np.float32))
        else:
            combined_mask = np.zeros((h, w), dtype=np.float32)

        # Calculate masked spike energy (explained by detections)
        masked_spike = spike_map * combined_mask
        masked_energy = float(np.sum(masked_spike))

        # Calculate residual (unexplained) energy
        residual_map = spike_map * (1 - combined_mask)
        residual_energy = float(np.sum(residual_map))

        # Calculate residue ratio
        residue_ratio = residual_energy / total_energy if total_energy > 0 else 0.0

        # Motion compensation: Check for camera jitter
        camera_motion, is_compensated = self._detect_camera_motion(spike_map)

        # Adjust residue if camera motion detected
        if is_compensated:
            adjusted_residue = max(0, residue_ratio - camera_motion)
        else:
            adjusted_residue = residue_ratio

        # Detect unexplained blobs
        unexplained_blobs = self._find_unexplained_blobs(residual_map)

        # Check for spatial anomaly (high residue = something moving outside masks)
        has_spatial_anomaly = adjusted_residue > self.residue_threshold

        # Check for volumetric anomaly (count outside expected range)
        min_v, max_v = n_volumetric_range
        has_volumetric_anomaly = False
        if max_v > 0:  # Only check if we have valid volumetric data
            has_volumetric_anomaly = n_visible < min_v or n_visible > max_v

        return FusionResult(
            residual_spike_energy=residual_energy,
            total_spike_energy=total_energy,
            residue_ratio=adjusted_residue,
            unexplained_blobs=unexplained_blobs,
            has_spatial_anomaly=has_spatial_anomaly,
            has_volumetric_anomaly=has_volumetric_anomaly,
            motion_compensated=is_compensated,
            camera_motion_energy=camera_motion,
        )

    def _detect_camera_motion(self, spike_map: np.ndarray) -> Tuple[float, bool]:
        """
        Detect global camera motion (jitter) from spike distribution.

        Returns:
            (motion_energy, is_camera_motion)
        """
        # Calculate global spike mean
        current_mean = float(np.mean(spike_map))

        if self.motion_state.prev_frame_mean is not None:
            # Calculate frame-to-frame difference
            motion_diff = abs(current_mean - self.motion_state.prev_frame_mean)

            # Update history
            self.motion_state.motion_history.append(motion_diff)
            if len(self.motion_state.motion_history) > self.motion_state.max_history:
                self.motion_state.motion_history.pop(0)

            # Check if this is camera motion (uniform global change)
            is_camera = motion_diff > self.motion_threshold
        else:
            motion_diff = 0.0
            is_camera = False

        self.motion_state.prev_frame_mean = current_mean

        return motion_diff, is_camera

    def _find_unexplained_blobs(self, residual_map: np.ndarray) -> List[Dict[str, Any]]:
        """
        Find connected regions in the residual map.

        Returns:
            List of blob dictionaries with position and energy.
        """
        try:
            import cv2  # pylint: disable=import-outside-toplevel

            # Threshold residual map
            threshold = 0.1  # 10% of max energy
            max_val = residual_map.max() if residual_map.max() > 0 else 1.0
            binary = (residual_map > max_val * threshold).astype(np.uint8)

            # Find contours
            contours, _ = cv2.findContours(  # pylint: disable=no-member
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,  # pylint: disable=no-member
            )

            blobs = []
            for contour in contours:
                area = cv2.contourArea(contour)  # pylint: disable=no-member
                if area >= self.min_blob_area:
                    moments = cv2.moments(contour)  # pylint: disable=no-member
                    if moments["m00"] > 0:
                        cx = int(moments["m10"] / moments["m00"])
                        cy = int(moments["m01"] / moments["m00"])

                        # Calculate blob energy
                        mask = np.zeros_like(residual_map, dtype=np.uint8)
                        cv2.drawContours(
                            mask, [contour], 0, 1, -1
                        )  # pylint: disable=no-member
                        energy = float(np.sum(residual_map * mask))

                        blobs.append(
                            {
                                "center": (cx, cy),
                                "area": area,
                                "energy": energy,
                                "contour": contour.tolist(),
                            }
                        )

            return blobs

        except ImportError:
            logger.warning("[FusionV2] OpenCV not available for blob detection")
            return []
        except Exception:  # pylint: disable=broad-except
            logger.warning("[FusionV2] Blob detection error")
            return []

    def reset_motion_state(self) -> None:
        """Reset motion tracking state (call on new video/session)."""
        self.motion_state = MotionState()


if __name__ == "__main__":
    # Quick test
    engine = FusionEngineV2()

    # Create mock data
    spike_map = np.random.rand(480, 640).astype(np.float32) * 0.5
    masks = [np.zeros((480, 640), dtype=np.uint8)]
    masks[0][100:200, 100:300] = 1  # Mock mask

    result = engine.fuse_spike_mask(
        spike_map=spike_map,
        masks=masks,
        n_visible=5,
        n_volumetric_range=(4, 6),
    )

    print(f"Residual Energy: {result.residual_spike_energy:.4f}")
    print(f"Residue Ratio: {result.residue_ratio:.4f}")
    print(f"Spatial Anomaly: {result.has_spatial_anomaly}")
    print(f"Volumetric Anomaly: {result.has_volumetric_anomaly}")
    print(f"Unexplained Blobs: {len(result.unexplained_blobs)}")
