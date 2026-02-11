"""
Unit Tests for Phase 3: Geometric Kernel

Verifies the functionality of:
1. AlphaHullWrapper (Volume Calculation, Golden Alpha Binary Search)
2. MathUtils (Volumetric Count Formulas)

CFG Structure:
Start Symbol: TestPhase3Geometry
"""

import unittest
import numpy as np
import os
import sys

# Ensure alphashape is in path
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../../Techs/alphashape-master/alphashape-master"
        )
    ),
)

from v2_logic.kernels.alphashape_wrapper import AlphaHullWrapper
from v2_logic.utils.math_utils import MathUtils


class TestPhase3Geometry(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.wrapper = AlphaHullWrapper()

        # Create a synthetic cube point cloud (1x1x1 = 1 cm³)
        cls.cube_points = cls._generate_cube_points(size=1.0, samples_per_edge=5)

        # Create a synthetic sphere point cloud
        cls.sphere_points = cls._generate_sphere_points(radius=1.0, num_samples=200)

    @staticmethod
    def _generate_cube_points(size=1.0, samples_per_edge=5):
        """Generate points on the surface of a cube."""
        points = []
        step = size / (samples_per_edge - 1)

        # Generate points on all 6 faces
        for i in range(samples_per_edge):
            for j in range(samples_per_edge):
                # Front and back faces
                points.append([i * step, j * step, 0])
                points.append([i * step, j * step, size])
                # Left and right faces
                points.append([0, i * step, j * step])
                points.append([size, i * step, j * step])
                # Top and bottom faces
                points.append([i * step, 0, j * step])
                points.append([i * step, size, j * step])

        return np.array(points)

    @staticmethod
    def _generate_sphere_points(radius=1.0, num_samples=200):
        """Generate points on the surface of a sphere."""
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle

        for i in range(num_samples):
            y = 1 - (i / float(num_samples - 1)) * 2  # y goes from 1 to -1
            r = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i

            x = np.cos(theta) * r
            z = np.sin(theta) * r

            points.append([x * radius, y * radius, z * radius])

        return np.array(points)

    def test_01_compute_hull_cube(self):
        """Test hull computation for a cube."""
        print("\n[Test] AlphaHull Volume (Cube)")

        hull = self.wrapper.compute_hull(self.cube_points, alpha=0.5)
        volume = hull.volume

        print(f"   Cube hull volume: {volume:.4f} cm³ (expected ~1.0)")

        self.assertIsNotNone(hull, "Hull should not be None")
        self.assertGreater(volume, 0.0, "Volume should be positive")

        # Allow some tolerance due to sampling
        self.assertAlmostEqual(volume, 1.0, delta=0.3)

    def test_02_compute_hull_sphere(self):
        """Test hull computation for a sphere."""
        print("\n[Test] AlphaHull Volume (Sphere)")

        hull = self.wrapper.compute_hull(self.sphere_points, alpha=1.0)
        volume = hull.volume

        # Expected volume of sphere with radius 1.0: (4/3)π ≈ 4.19
        expected_volume = (4 / 3) * np.pi * (1.0**3)

        print(
            f"   Sphere hull volume: {volume:.4f} cm³ (expected ~{expected_volume:.4f})"
        )

        self.assertIsNotNone(hull, "Hull should not be None")
        self.assertGreater(volume, 0.0, "Volume should be positive")

        # Sphere approximation can have significant error
        self.assertAlmostEqual(volume, expected_volume, delta=1.0)

    def test_03_golden_alpha_search(self):
        """Test Golden Alpha binary search."""
        print("\n[Test] Golden Alpha Binary Search")

        # Known target: cube with volume 1.0 cm³
        target_volume = 1.0

        golden_alpha = self.wrapper.find_golden_alpha(
            self.cube_points, target_volume, tolerance=0.1, max_iter=20
        )

        print(f"   Found Golden Alpha: {golden_alpha:.4f}")

        # Verify the found alpha produces correct volume
        hull = self.wrapper.compute_hull(self.cube_points, golden_alpha)
        actual_volume = hull.volume

        error = abs(actual_volume - target_volume) / target_volume
        print(f"   Verification: V={actual_volume:.4f}, error={error*100:.2f}%")

        self.assertLess(error, 0.15, "Golden Alpha should converge within tolerance")

    def test_04_volumetric_count_formula(self):
        """Test V3.1 volumetric count formula."""
        print("\n[Test] Volumetric Count Formula")

        # Example: Stack of 10 cups, each 350 cm³, with 80% packing efficiency
        v_unit = 350.0
        n_expected = 10
        rho = 0.8
        v_stack = (n_expected * v_unit) / rho  # Reverse calculation

        n_vol = MathUtils.calculate_volumetric_count(v_stack, rho, v_unit)

        print(f"   Input: V_stack={v_stack:.2f}, ρ={rho}, V_unit={v_unit}")
        print(f"   N_vol = {n_vol:.2f} (expected ~{n_expected})")

        self.assertAlmostEqual(n_vol, n_expected, delta=0.5)

    def test_05_estimate_stack_efficiency(self):
        """Test inverse efficiency estimation."""
        print("\n[Test] Stack Efficiency Estimation")

        # Known scenario
        n_visible = 8
        v_unit = 250.0
        v_stack = 3000.0

        rho = MathUtils.estimate_stack_efficiency(n_visible, v_stack, v_unit)

        # Expected: (8 × 250) / 3000 = 0.667
        expected_rho = (n_visible * v_unit) / v_stack

        print(f"   N_visible={n_visible}, V_stack={v_stack}, V_unit={v_unit}")
        print(f"   Estimated ρ = {rho:.3f} (expected {expected_rho:.3f})")

        self.assertAlmostEqual(rho, expected_rho, delta=0.01)
        self.assertLessEqual(rho, 1.0, "Efficiency should be capped at 1.0")

    def test_06_edge_case_zero_volume(self):
        """Test edge case: zero unit volume."""
        print("\n[Test] Edge Case: Zero V_unit")

        n_vol = MathUtils.calculate_volumetric_count(1000.0, 0.8, 0.0)
        self.assertEqual(n_vol, 0.0, "Should return 0 for zero V_unit")


if __name__ == "__main__":
    unittest.main()
