"""
Fusion Engine V2 (Multi-Shield Validation)

Fuses V2E spike data with SAM2 masks and volumetric analysis
to detect anomalies through a Multi-Shield Architecture.
Core component of the Recursive Intent "Triple Check" system.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : FusionEngineV2 (this module)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <FusionEngineV2>  → Main fusion orchestrator                            │
  │  <FusionResult>    → Output dataclass with residue and anomaly info      │
  │  <ShieldResult>    → Per-shield verdict (confidence + detail)            │
  │  <MotionState>     → Camera jitter tracking state                        │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <PerceptionState>  ← from types.graph_state (State container)           │
  │  <AlphaHullWrapper> ← from kernels.alphashape_wrapper (Volume calc)      │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : numpy.ndarray, float, int, List, Dict

Production Rules:
  FusionEngineV2 → __init__ + fuse_spike_mask + run_shields
                 + suppress_contra_intents
  run_shields    → shield_spatial + shield_volumetric + shield_latent
                 → weighted_sum → fusion_confidence
  fuse_spike_mask → suppress_contras + subtract_masks + calculate_residue
                 → FusionResult
═══════════════════════════════════════════════════════════════════════════════

Pattern: Strategy + Chain of Responsibility
- Each Shield encapsulates one validation dimension.
- Shields are evaluated in order; aggregate confidence drives decisions.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ShieldResult:
    """Result from a single validation shield."""

    name: str  # Shield identifier
    confidence: float  # 0.0 (fail) to 1.0 (pass)
    passed: bool  # Whether this shield passed
    details: Dict[str, Any] = field(default_factory=dict)  # Diagnostic info


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

    # Phase 4: Multi-Shield Scores
    shield_scores: Dict[str, float] = field(default_factory=dict)
    fusion_confidence: float = 1.0  # Aggregate confidence (0-1)


@dataclass
class MotionState:
    """Tracks motion history for camera jitter detection."""

    prev_frame_mean: Optional[float] = None
    motion_history: List[float] = field(default_factory=list)
    max_history: int = 10


class FusionEngineV2:
    """
    Multi-Shield Fusion Engine for perception validation.

    Implements:
    1. Shield 1 (Spatial): spike_map - (spike_map * combined_mask)
    2. Shield 2 (Volumetric): N_visible vs AlphaHull volume estimate
    3. Shield 3 (Latent): Identity stability via V-JEPA similarity

    Pattern: Strategy + Chain of Responsibility
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

        # Phase 4: Shield weights (must sum to 1.0)
        self.shield_weights = {
            "spatial": 0.5,
            "volumetric": 0.3,
            "latent": 0.2,
        }
        self.confidence_threshold = 0.6  # Below this → SLM Audit

        # V3.3: Negative mask storage for Contra Intent suppression
        self._negative_masks: List[np.ndarray] = []
        self._negative_mask_dilation_px: int = 5  # Prevent geometric leakage

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
            n_visible: Visual count from CountVid.
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

        # V3.3: Subtract negative masks (Contra Intent suppression)
        # This prevents distractors from triggering anomaly alarms
        contra_mask = self._build_combined_negative_mask(h, w)
        if contra_mask is not None:
            # Energy inside contra masks is "explained" (suppressed)
            spike_map = spike_map * (1 - contra_mask)
            logger.debug(
                "[FusionV2] Contra mask applied: suppressed %.2f%% of spike energy",
                float(np.sum(contra_mask)) / (h * w) * 100,
            )

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
                cv2.RETR_EXTERNAL,  # pylint: disable=no-member
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
                        cv2.drawContours(  # pylint: disable=no-member
                            mask, [contour], 0, 1, -1
                        )
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

    # =========================================================================
    # Phase 4: Multi-Shield Architecture
    # =========================================================================

    def shield_spatial(self, fusion_result: FusionResult) -> ShieldResult:
        """
        Shield 1 (Spatio-Temporal): Evaluate spike-mask residue.
        High residue means something is moving outside detected masks.

        Returns:
            ShieldResult with spatial confidence.
        """
        # Confidence = 1 - residue_ratio (clamped to 0-1)
        confidence = max(0.0, 1.0 - fusion_result.residue_ratio)
        passed = not fusion_result.has_spatial_anomaly

        return ShieldResult(
            name="spatial",
            confidence=confidence,
            passed=passed,
            details={
                "residue_ratio": fusion_result.residue_ratio,
                "threshold": self.residue_threshold,
                "blob_count": len(fusion_result.unexplained_blobs),
            },
        )

    def shield_volumetric(
        self,
        n_visible: int,
        n_volumetric_range: Tuple[int, int],
        v_stack: float = 0.0,
        v_unit: float = 0.0,
    ) -> ShieldResult:
        """
        Shield 2 (Volumetric): Check if the visual count is physically
        plausible given the measured volume.

        Args:
            n_visible: Count from CountVid.
            n_volumetric_range: (min, max) from volumetric estimation.
            v_stack: Total observed stack volume (m³).
            v_unit: Known unit volume (m³).

        Returns:
            ShieldResult with volumetric confidence.
        """
        min_v, max_v = n_volumetric_range

        if max_v <= 0:
            # No volumetric data available; shield is neutral
            return ShieldResult(
                name="volumetric",
                confidence=0.5,
                passed=True,
                details={"reason": "no_volumetric_data"},
            )

        # How far is n_visible from the valid range?
        if min_v <= n_visible <= max_v:
            confidence = 1.0
        else:
            range_span = max(1, max_v - min_v)
            if n_visible < min_v:
                distance = min_v - n_visible
            else:
                distance = n_visible - max_v
            # Confidence decays with distance from valid range
            confidence = max(0.0, 1.0 - (distance / range_span))

        passed = min_v <= n_visible <= max_v

        return ShieldResult(
            name="volumetric",
            confidence=confidence,
            passed=passed,
            details={
                "n_visible": n_visible,
                "range": (min_v, max_v),
                "v_stack": v_stack,
                "v_unit": v_unit,
            },
        )

    def shield_latent(
        self,
        track_similarities: Optional[List[float]] = None,
    ) -> ShieldResult:
        """
        Shield 3 (Latent Identity): Evaluate identity stability using
        V-JEPA / ReID latent similarity scores.

        A high mean similarity means objects are consistently tracked.
        Low similarity suggests identity confusion or new objects.

        Args:
            track_similarities: List of cosine similarities for each track.
                                None if V-JEPA data is unavailable.

        Returns:
            ShieldResult with latent confidence.
        """
        if not track_similarities or len(track_similarities) == 0:
            # No latent data; shield is neutral
            return ShieldResult(
                name="latent",
                confidence=0.5,
                passed=True,
                details={"reason": "no_latent_data"},
            )

        mean_similarity = float(np.mean(track_similarities))
        # Clamp similarity to 0-1 range as confidence
        confidence = max(0.0, min(1.0, mean_similarity))
        passed = confidence >= 0.4  # Lenient threshold for latent

        return ShieldResult(
            name="latent",
            confidence=confidence,
            passed=passed,
            details={
                "mean_similarity": mean_similarity,
                "num_tracks": len(track_similarities),
                "min_similarity": float(np.min(track_similarities)),
            },
        )

    def run_shields(
        self,
        fusion_result: FusionResult,
        n_visible: int,
        n_volumetric_range: Tuple[int, int],
        v_stack: float = 0.0,
        v_unit: float = 0.0,
        track_similarities: Optional[List[float]] = None,
    ) -> FusionResult:
        """
        Run all three shields and compute aggregate fusion_confidence.
        Updates the FusionResult in-place with shield scores.

        Args:
            fusion_result: Result from fuse_spike_mask.
            n_visible: Visual count from CountVid.
            n_volumetric_range: Volume-estimated count range.
            v_stack: Total stack volume.
            v_unit: Unit object volume.
            track_similarities: ReID/V-JEPA similarity list.

        Returns:
            Updated FusionResult with shield_scores and fusion_confidence.
        """
        # Run each shield
        s1 = self.shield_spatial(fusion_result)
        s2 = self.shield_volumetric(n_visible, n_volumetric_range, v_stack, v_unit)
        s3 = self.shield_latent(track_similarities)

        shields = [s1, s2, s3]

        # Compute weighted confidence
        fusion_confidence = sum(
            self.shield_weights[s.name] * s.confidence for s in shields
        )

        # Store results
        fusion_result.shield_scores = {s.name: s.confidence for s in shields}
        fusion_result.fusion_confidence = fusion_confidence

        # Log shield verdicts
        shield_summary = " | ".join(
            f"{s.name}: {'✓' if s.passed else '✗'} ({s.confidence:.2f})"
            for s in shields
        )
        logger.info(
            "[FusionV2] Shields: %s → confidence=%.3f (threshold=%.2f)",
            shield_summary,
            fusion_confidence,
            self.confidence_threshold,
        )

        if fusion_confidence < self.confidence_threshold:
            logger.warning(
                "[FusionV2] LOW CONFIDENCE (%.3f) — SLM Audit recommended.",
                fusion_confidence,
            )

        return fusion_result

    def suppress_contra_intents(
        self,
        negative_masks: List[np.ndarray],
    ) -> None:
        """
        V3.3 Step 12.2: Register Contra Intent negative masks for permanent
        suppression. Once registered, energy inside these masks will never
        trigger anomaly alarms.

        Args:
            negative_masks: List of binary masks (H, W) for each distractor.
                           Typically generated from SAM2 segmenting the contra object.
        """
        for mask in negative_masks:
            if mask is not None and mask.ndim >= 2:
                self._negative_masks.append(mask.astype(np.float32))

        logger.info(
            "[FusionV2] Registered %d negative masks (total: %d)",
            len(negative_masks),
            len(self._negative_masks),
        )

    def _build_combined_negative_mask(
        self,
        target_h: int,
        target_w: int,
    ) -> Optional[np.ndarray]:
        """
        Combine all registered negative masks into one, with dilation
        to prevent geometric leakage at mask boundaries.

        Args:
            target_h: Target height for resizing.
            target_w: Target width for resizing.

        Returns:
            Combined binary mask (H, W) or None if no negative masks.
        """
        if not self._negative_masks:
            return None

        import cv2  # pylint: disable=import-outside-toplevel

        combined = np.zeros((target_h, target_w), dtype=np.float32)

        for mask in self._negative_masks:
            # Resize mask if dimensions don't match
            if mask.shape[:2] != (target_h, target_w):
                mask_resized = cv2.resize(
                    mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
                )
            else:
                mask_resized = mask

            combined = np.maximum(combined, mask_resized)

        # Apply dilation to prevent leakage at boundaries
        dilation_px = self._negative_mask_dilation_px
        if dilation_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1)
            )
            combined = cv2.dilate(combined, kernel, iterations=1)

        # Clip to binary [0, 1]
        combined = np.clip(combined, 0.0, 1.0)

        return combined


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

    # Run Multi-Shield validation
    result = engine.run_shields(
        fusion_result=result,
        n_visible=5,
        n_volumetric_range=(4, 6),
        track_similarities=[0.85, 0.92, 0.78],
    )

    print(f"Residual Energy: {result.residual_spike_energy:.4f}")
    print(f"Residue Ratio: {result.residue_ratio:.4f}")
    print(f"Spatial Anomaly: {result.has_spatial_anomaly}")
    print(f"Volumetric Anomaly: {result.has_volumetric_anomaly}")
    print(f"Shield Scores: {result.shield_scores}")
    print(f"Fusion Confidence: {result.fusion_confidence:.3f}")
