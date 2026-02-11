"""
Unit Tests for Phase 2: Physical Density Engine

Verifies the functionality of:
1. DINOv2Engine (Feature Extraction, Specularity Analysis)
2. DensityPredictor (MLP Regressor, Calibration, Persistence)

CFG Structure:
Start Symbol: TestPhase2Density
"""

import unittest
import torch
import numpy as np
import os
import shutil
from PIL import Image
from v2_logic.models.dinov2_engine import DINOv2Engine
from v2_logic.models.density_predictor import DensityPredictor


class TestPhase2Density(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_dir = "v2_logic/tests/temp_density_test"
        os.makedirs(cls.test_dir, exist_ok=True)

        # Create a dummy image (random noise)
        cls.dummy_image = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )

        # Initialize engines
        # Use CPU for testing to avoid CUDA OOM if another process is using it,
        # but DINOv2 defaults to CUDA if available.
        cls.dino_engine = DINOv2Engine(device="cpu")
        cls.density_predictor = DensityPredictor()

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_01_dinov2_extraction(self):
        """Test DINOv2 feature extraction shape."""
        print("\n[Test] DINOv2 Feature Extraction")
        features = self.dino_engine.extract_features(self.dummy_image)
        self.assertEqual(features.shape, (768,), "Feature vector should be (768,)")
        self.assertTrue(torch.is_tensor(features), "Output should be a tensor")

    def test_02_specularity_analysis(self):
        """Test Specularity Analysis (Visual Complexity)."""
        print("\n[Test] Specularity Analysis")
        score = self.dino_engine.analyze_specularity(self.dummy_image)
        print(f"   Specularity Score (Random Noise): {score:.4f}")
        self.assertIsInstance(score, float, "Score should be a float")
        self.assertGreater(score, 0.0, "Variance should be positive")

        # Create a flat image (low complexity)
        flat_image = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 128)
        flat_score = self.dino_engine.analyze_specularity(flat_image)
        print(f"   Specularity Score (Flat Image): {flat_score:.4f}")

        # Random noise should have higher variance/complexity than flat image
        # Note: DINOv2 features for flat image might still have some variance due to patch pos encoding,
        # but noise should be higher.
        self.assertGreater(
            score, flat_score, "Noise should be more complex than flat image"
        )

    def test_03_density_predictor_heuristic(self):
        """Test Density Predictor Heuristic Calibration."""
        print("\n[Test] Density Predictor Heuristic")
        self.assertFalse(
            self.density_predictor.is_fitted, "Should not be fitted initially"
        )

        self.density_predictor.calibrate_heuristic()
        self.assertTrue(
            self.density_predictor.is_fitted, "Should be fitted after calibration"
        )

        # Test prediction
        features = torch.randn(768).numpy()
        density = self.density_predictor.predict(features.reshape(1, -1))
        print(f"   Predicted Density: {density[0]:.4f}")
        self.assertEqual(density.shape, (1,), "Output should be (1,)")

        # Basic check: is result somewhat reasonable?
        # Since we trained on synthetic data 0.1-20.0, it should be in that ballpark roughly
        # but specific values depend on the random features.

    def test_04_persistence(self):
        """Test Save and Load."""
        print("\n[Test] Density Predictor Persistence")
        save_path = os.path.join(self.test_dir, "density_model.pkl")

        self.density_predictor.calibrate_heuristic()  # Ensure fitted
        original_pred = self.density_predictor.predict(np.zeros((1, 768)))

        self.density_predictor.save(save_path)
        self.assertTrue(os.path.exists(save_path), "Model file should exist")

        # Load into new instance
        new_predictor = DensityPredictor()
        new_predictor.load(save_path)

        new_pred = new_predictor.predict(np.zeros((1, 768)))

        np.testing.assert_array_almost_equal(
            original_pred,
            new_pred,
            decimal=5,
            err_msg="Loaded model predictions should match original",
        )


if __name__ == "__main__":
    unittest.main()
