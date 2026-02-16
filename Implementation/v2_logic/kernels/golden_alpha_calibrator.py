"""
Golden Alpha Calibrator (Step 3.5)

Dedicated calibration module that finds the optimal Alpha parameter
for the Concave Hull such that the calculated volume matches the
SLM-estimated unit volume (V_unit).

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : GoldenAlphaCalibratorModule (this file)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <GoldenAlphaCalibrator>  → Main calibration engine                       │
  │  <CalibrationResult>      → Output dataclass with alpha + metadata        │
  │  <_select_isolated>       → Cluster isolation selector                    │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <AlphaHullWrapper>  ← from kernels.alphashape_wrapper (Volume calc)      │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : float, int, numpy.ndarray, bool

Production Rules:
  GoldenAlphaCalibratorModule → imports + <CalibrationResult>
                              + <GoldenAlphaCalibrator>
  GoldenAlphaCalibrator       → __init__ + calibrate + _select_isolated
                              + _binary_search_alpha
  calibrate → _select_isolated + _binary_search_alpha → CalibrationResult
═══════════════════════════════════════════════════════════════════════════════

Pattern: Strategy
- Encapsulates the alpha-search algorithm as a replaceable strategy.
- Can be swapped for different search methods (binary, golden-section, etc.).
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of Golden Alpha calibration."""

    golden_alpha: float  # The optimal alpha parameter
    v_calculated: float  # Volume calculated at golden_alpha (m³)
    v_target: float  # Target unit volume from SLM (m³)
    error_pct: float  # Percentage error |v_calc - v_target| / v_target
    iterations: int  # Number of binary search iterations
    is_converged: bool  # Whether calibration converged within tolerance
    cluster_index: int  # Index of the cluster used for calibration


class GoldenAlphaCalibrator:
    """
    Finds the optimal Alpha for Concave Hull calibration.

    The Golden Alpha is the value of alpha where:
        |V_concave(alpha) - V_unit| < tolerance

    This replaces the inline calibration logic previously in v3_math_node.

    Pattern: Strategy
    - Binary search for alpha parameter.
    """

    def __init__(
        self,
        tolerance: float = 0.05,
        max_iterations: int = 50,
        alpha_range: Tuple[float, float] = (0.01, 10.0),
    ):
        """
        Args:
            tolerance: Convergence tolerance (5% = 0.05 default).
            max_iterations: Maximum binary search iterations.
            alpha_range: (min_alpha, max_alpha) search bounds.
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.alpha_min, self.alpha_max = alpha_range

    def calibrate(
        self,
        clusters: List[Dict[str, Any]],
        v_unit: float,
        alpha_wrapper: Any,
    ) -> Optional[CalibrationResult]:
        """
        Calibrate Golden Alpha using the most isolated cluster.

        Args:
            clusters: List of cluster dicts from sam2_depth_node.
                      Each must have 'points' (Nx3 ndarray) and optionally 'volume_m3'.
            v_unit: Target unit volume in m³ from SLM.
            alpha_wrapper: AlphaHullWrapper instance.

        Returns:
            CalibrationResult or None if calibration fails.
        """
        if not clusters or v_unit <= 0:
            logger.warning(
                "[GoldenAlpha] Cannot calibrate: clusters=%d, v_unit=%.6f",
                len(clusters),
                v_unit,
            )
            return None

        # Step 1: Select the most isolated cluster (best for calibration)
        cluster_idx, points = self._select_isolated_cluster(clusters)
        if points is None or len(points) < 4:
            logger.warning(
                "[GoldenAlpha] Selected cluster has too few points (%d < 4)",
                len(points) if points is not None else 0,
            )
            return None

        logger.info(
            "[GoldenAlpha] Calibrating with cluster %d (%d points), target V=%.6f m³",
            cluster_idx,
            len(points),
            v_unit,
        )

        # Step 2: Binary search for optimal alpha
        result = self._binary_search_alpha(points, v_unit, alpha_wrapper)
        if result is not None:
            result.cluster_index = cluster_idx

        return result

    def calibrate_per_intent(
        self,
        clusters: List[Dict[str, Any]],
        v_units: Dict[str, float],
        alpha_wrapper: Any,
    ) -> Dict[str, CalibrationResult]:
        """
        Calibrate Golden Alpha per intent type (multi-SKU support).

        Args:
            clusters: Cluster list with optional 'label' field.
            v_units: Dict mapping label → V_unit.
            alpha_wrapper: AlphaHullWrapper instance.

        Returns:
            Dict mapping label → CalibrationResult.
        """
        results = {}
        for label, v_unit in v_units.items():
            # Find clusters matching this label
            matching = [
                c for c in clusters if c.get("label", "").lower() == label.lower()
            ]
            if not matching:
                # Try using any available cluster as fallback
                matching = clusters

            result = self.calibrate(matching, v_unit, alpha_wrapper)
            if result is not None:
                results[label] = result
                logger.info(
                    "[GoldenAlpha] '%s' → alpha=%.4f (error=%.1f%%)",
                    label,
                    result.golden_alpha,
                    result.error_pct * 100,
                )

        return results

    def _select_isolated_cluster(
        self,
        clusters: List[Dict[str, Any]],
    ) -> Tuple[int, Optional[np.ndarray]]:
        """
        Select the most isolated cluster for calibration accuracy.

        Strategy: Pick the cluster with fewest neighboring clusters
        (measured by bounding box overlap). Falls back to the
        topmost cluster (highest Y position — likely most visible).

        Returns:
            (cluster_index, points_ndarray) or (-1, None).
        """
        if not clusters:
            return -1, None

        if len(clusters) == 1:
            points = clusters[0].get("points")
            if points is not None:
                return 0, (
                    np.array(points) if not isinstance(points, np.ndarray) else points
                )
            return -1, None

        # Score each cluster by isolation (inverse of overlap count)
        scores = []
        for idx, cluster in enumerate(clusters):
            bbox_a = cluster.get("bbox", {})
            if not bbox_a or bbox_a.get("w", 0) == 0:
                scores.append(-1)
                continue

            overlap_count = 0
            for other_idx, other in enumerate(clusters):
                if other_idx == idx:
                    continue
                bbox_b = other.get("bbox", {})
                if not bbox_b or bbox_b.get("w", 0) == 0:
                    continue

                # Simple overlap check
                ax1 = bbox_a.get("x", 0)
                ay1 = bbox_a.get("y", 0)
                ax2 = ax1 + bbox_a.get("w", 0)
                ay2 = ay1 + bbox_a.get("h", 0)
                bx1 = bbox_b.get("x", 0)
                by1 = bbox_b.get("y", 0)
                bx2 = bx1 + bbox_b.get("w", 0)
                by2 = by1 + bbox_b.get("h", 0)

                if ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1:
                    overlap_count += 1

            scores.append(-overlap_count)  # Negative = less overlap = better

        # Select best (least overlap)
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        points = clusters[best_idx].get("points")
        if points is not None:
            points = np.array(points) if not isinstance(points, np.ndarray) else points
            return best_idx, points

        return -1, None

    def _binary_search_alpha(
        self,
        points: np.ndarray,
        v_target: float,
        alpha_wrapper: Any,
    ) -> Optional[CalibrationResult]:
        """
        Binary search for alpha where V_concave(alpha) ≈ V_target.

        A smaller alpha produces a tighter hull (larger volume reduction).
        A larger alpha produces a looser hull (closer to convex hull).

        Returns:
            CalibrationResult or None.
        """
        lo = self.alpha_min
        hi = self.alpha_max

        best_alpha = (lo + hi) / 2.0
        best_error = float("inf")
        best_v_calc = 0.0

        for iteration in range(self.max_iterations):
            mid = (lo + hi) / 2.0

            try:
                v_calc = alpha_wrapper.compute_volume_at_alpha(points, mid)
            except Exception as exc:
                logger.warning(
                    "[GoldenAlpha] Volume calc failed at alpha=%.4f: %s", mid, exc
                )
                # Try to continue with a shifted range
                lo = mid
                continue

            if v_calc <= 0:
                # Alpha too tight, loosen it
                lo = mid
                continue

            error = abs(v_calc - v_target) / v_target if v_target > 0 else float("inf")

            if error < best_error:
                best_error = error
                best_alpha = mid
                best_v_calc = v_calc

            # Check convergence
            if error < self.tolerance:
                logger.info(
                    "[GoldenAlpha] Converged after %d iterations: "
                    "alpha=%.4f, V_calc=%.6f, V_target=%.6f, error=%.2f%%",
                    iteration + 1,
                    mid,
                    v_calc,
                    v_target,
                    error * 100,
                )
                return CalibrationResult(
                    golden_alpha=mid,
                    v_calculated=v_calc,
                    v_target=v_target,
                    error_pct=error,
                    iterations=iteration + 1,
                    is_converged=True,
                    cluster_index=-1,
                )

            # Binary search direction
            if v_calc > v_target:
                # Hull too big → tighten (smaller alpha)
                hi = mid
            else:
                # Hull too small → loosen (larger alpha)
                lo = mid

        # Did not converge but return best result
        logger.warning(
            "[GoldenAlpha] Did not converge after %d iterations. "
            "Best: alpha=%.4f, error=%.2f%%",
            self.max_iterations,
            best_alpha,
            best_error * 100,
        )
        return CalibrationResult(
            golden_alpha=best_alpha,
            v_calculated=best_v_calc,
            v_target=v_target,
            error_pct=best_error,
            iterations=self.max_iterations,
            is_converged=False,
            cluster_index=-1,
        )
