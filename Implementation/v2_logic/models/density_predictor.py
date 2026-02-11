"""
Density Predictor (Phase 2)

Wraps sklearn's MLPRegressor to predict physical density ($\rho$) from DINOv2 semantic features.
Includes functionality for online calibration and heuristic initialization.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : DensityPredictor (this module)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <DensityPredictor> → __init__ | fit | predict | save | load | calibrate_heuristic
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <MLPRegressor>  ← from sklearn.neural_network                            │
  │  <joblib>        ← import joblib (persistence)                            │
  │  <np>            ← import numpy                                           │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : "lbfgs", "relu", (100, 50), 1e-4

Production Rules:
  DensityPredictor  → imports + <DensityPredictor>
  <DensityPredictor>→ class DensityPredictor: <Methods>+
  <Methods>         → __init__()
                    | fit(X, y)
                    | predict(X) -> float
                    | save(path)
                    | load(path)
                    | calibrate_heuristic()
═══════════════════════════════════════════════════════════════════════════════
"""

from sklearn.neural_network import MLPRegressor
import numpy as np
import joblib
import os


class DensityPredictor:
    """
    Predicts physical density from visual features using a lightweight MLP.

    Pattern: Adapter / Strategy
    - wraps sklearn API into a domain-specific interface for density prediction.
    """

    def __init__(
        self,
        hidden_layer_sizes=(100, 50),
        activation="relu",
        solver="lbfgs",
        alpha=1e-4,
        random_state=42,
    ):
        """
        Initialize the MLP Regressor with physics-informed defaults.

        Args:
            hidden_layer_sizes: Architecture of the MLP.
            activation: Activation function ('relu' is standard).
            solver: Optimizer ('lbfgs' is best for small datasets/calibration).
            alpha: L2 regularization term.
        """
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            random_state=random_state,
            max_iter=1000,  # Allow convergence for lbfgs
        )
        self.is_fitted = False

    def fit(self, X, y):
        """
        Train the density predictor on calibration data.

        Args:
            X (np.ndarray): DINOv2 feature vectors (N, 768).
            y (np.ndarray): Target density values (N,).
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Predict density from features.

        Args:
            X (np.ndarray): Feature vector (768,) or batch (N, 768).

        Returns:
            np.ndarray: Predicted density values.
        """
        if not self.is_fitted:
            # If not fitted, return a neutral density (e.g. water = 1.0)
            # or raise a warning. For safety, we'll return 1.0s.
            print(
                "[DensityPredictor] Warning: Model not fitted. Returning default density 1.0."
            )
            if X.ndim == 1:
                return np.array([1.0])
            return np.ones(X.shape[0])

        return self.model.predict(X)

    def calibrate_heuristic(self):
        """
        Initializes the model with a heuristic "prioir":
        Higher complexity/variance (simulated in features) -> Higher density.

        This acts as a 'cold start' calibration so the model isn't random.
        """
        print(
            "[DensityPredictor] Calibrating with heuristic prior (Variance -> Density)..."
        )

        # Synthetic data generation
        # Let's assume higher norm/variance of feature vector correlates with density for this heuristic
        # (This is a simplified assumption for initialization)

        n_samples = 100
        input_dim = 768

        # Generate random vectors with varying magnitudes/variances
        X_synthetic = np.random.randn(n_samples, input_dim)

        # Heuristic: Density = sigmoid(variance(features)) scaled to [0.5, 20.0]
        # Calculate variance of each sample
        variances = np.var(X_synthetic, axis=1)

        # Normalize variances to 0-1
        norm_var = (variances - variances.min()) / (
            variances.max() - variances.min() + 1e-6
        )

        # Map to density range [0.1 (aerogel) - 20.0 (gold/tungsten)]
        # Heuristic: linear mapping
        y_synthetic = 0.1 + (norm_var * 19.9)

        self.fit(X_synthetic, y_synthetic)
        print("[DensityPredictor] Heuristic calibration complete.")

    def save(self, path):
        """Save model to disk."""
        joblib.dump(self.model, path)
        print(f"[DensityPredictor] Model saved to {path}")

    def load(self, path):
        """Load model from disk."""
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.is_fitted = True
            print(f"[DensityPredictor] Model loaded from {path}")
        else:
            print(f"[DensityPredictor] Warning: Checkpoint {path} not found.")
