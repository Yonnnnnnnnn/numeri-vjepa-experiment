"""
AlphaShape Wrapper (Phase 3)

Wraps the `alphashape` library to compute 3D concave hulls from point clouds
and provides a method to find the "Golden Alpha" parameter via binary search
to match a target volume.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : AlphaHullWrapper (this module)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <AlphaHullWrapper>  → __init__ | compute_hull | find_golden_alpha        │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <alphashape>     ← from alphashape (3D hull generation)                  │
  │  <trimesh>        ← trimesh (mesh volume calculation)                     │
  │  <np>             ← import numpy                                          │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : "volume", alpha (float)

Production Rules:
  AlphaHullWrapper  → imports + <AlphaHullWrapper>
  <AlphaHullWrapper>→ class AlphaHullWrapper: <Methods>+
  <Methods>         → __init__()
                    | compute_hull(points, alpha) -> trimesh.Trimesh
                    | find_golden_alpha(points, target_volume, tolerance) -> float
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import os
import numpy as np

# Add alphashape to path
alphashape_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "../../Techs/alphashape-master/alphashape-master"
    )
)
if alphashape_path not in sys.path:
    sys.path.insert(0, alphashape_path)

import alphashape
import trimesh


class AlphaHullWrapper:
    """
    Wraps `alphashape` library for 3D volumetric analysis.

    Pattern: Adapter
    - Adapts the alphashape + trimesh interface to our volumetric needs.
    - Provides a binary search method to find the "Golden Alpha" parameter.
    """

    def __init__(self):
        """Initialize the AlphaHullWrapper."""
        pass

    def compute_hull(self, points: np.ndarray, alpha: float = 1.0) -> trimesh.Trimesh:
        """
        Computes the alpha shape (concave hull) for a 3D point cloud.

        Args:
            points (np.ndarray): (N, 3) array of 3D points.
            alpha (float): Alpha parameter. Higher = tighter hull.

        Returns:
            trimesh.Trimesh: The 3D mesh hull. Access volume via `.volume`.
        """
        if points.shape[0] < 4:
            raise ValueError("Need at least 4 points for 3D hull")

        hull = alphashape.alphashape(points, alpha)

        if not isinstance(hull, trimesh.Trimesh):
            # If alpha is too high, alphashape returns empty geometry
            raise ValueError(f"Alpha {alpha} produced invalid hull (likely too tight)")

        return hull

    def find_golden_alpha(
        self,
        points: np.ndarray,
        target_volume: float,
        tolerance: float = 0.05,
        max_iter: int = 50,
    ) -> float:
        """
        Binary search to find the alpha parameter that yields a hull with
        volume ≈ target_volume.

        This is the "Golden Alpha Calibration" for V3.1: we solve for the alpha
        that wraps a single object to match its known physical volume.

        Args:
            points (np.ndarray): (N, 3) point cloud of a single object.
            target_volume (float): Known physical volume (e.g., 350 cm³ for a cup).
            tolerance (float): Acceptable % error (default 5%).
            max_iter (int): Maximum binary search iterations.

        Returns:
            float: The alpha parameter that produces volume ≈ target_volume.
        """
        print(
            f"[AlphaHullWrapper] Searching for Golden Alpha (target: {target_volume:.2f} cm³)..."
        )

        # Alpha bounds:
        # - Lower bound (0.0): Convex hull (largest possible volume)
        # - Upper bound: Start with a high value, adjust if necessary
        alpha_min = 0.0
        alpha_max = 10.0

        # Check if convex hull is too small
        try:
            convex_hull = self.compute_hull(points, alpha_min)
            convex_volume = convex_hull.volume

            if convex_volume < target_volume * (1 - tolerance):
                print(
                    f"[AlphaHullWrapper] Warning: Convex hull ({convex_volume:.2f}) < target. Returning alpha=0.0"
                )
                return alpha_min
        except Exception as e:
            print(
                f"[AlphaHullWrapper] Error computing convex hull: {e}. Using fallback."
            )
            return 1.0

        # Binary search
        for i in range(max_iter):
            alpha_mid = (alpha_min + alpha_max) / 2.0

            try:
                hull = self.compute_hull(points, alpha_mid)
                current_volume = hull.volume
            except ValueError:
                # Alpha too high, hull collapsed
                alpha_max = alpha_mid
                continue

            error = abs(current_volume - target_volume) / target_volume

            print(
                f"  Iter {i+1}: alpha={alpha_mid:.4f}, V={current_volume:.2f}, error={error*100:.2f}%"
            )

            if error < tolerance:
                print(f"[AlphaHullWrapper] Converged! Golden Alpha = {alpha_mid:.4f}")
                return alpha_mid

            # Adjust bounds
            if current_volume > target_volume:
                # Hull is too large, increase alpha (tighter fit)
                alpha_min = alpha_mid
            else:
                # Hull is too small, decrease alpha (looser fit)
                alpha_max = alpha_mid

        # If max iterations reached, return best guess
        best_alpha = (alpha_min + alpha_max) / 2.0
        print(
            f"[AlphaHullWrapper] Max iterations reached. Best alpha = {best_alpha:.4f}"
        )
        return best_alpha
