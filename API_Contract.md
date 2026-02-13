"""
API Contract

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol : APIContract (this document)

Non-Terminals :
┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
│ <EngineAPI> → VJEPAEngine | Director | Executor │
│ <LoopAPI> → Feedback mechanisms │
└───────────────────────────────────────────────────────────────────────────┘

Terminals : str, float, int, Tensor, Latent

Production Rules:
APIContract → <EngineAPI>+ <LoopAPI>\*
═══════════════════════════════════════════════════════════════════════════════
"""

# API Contract: V-JEPA Event-Based Intelligence

This document defines the interfaces between the core modules of the Antigravity V2 system.

## 1. Engine & Kernel Interfaces

### 1.1. Density Engine (`DINOv2Engine` & `DensityPredictor`)

- **`analyze_specularity(image: np.ndarray) -> float`**
  - Input: Cropped object frame.
  - Output: Statistical variance coefficient (S).
- **`predict(features: np.ndarray) -> float`**
  - Input: DINOv2 latent vector + S.
  - Output: Predicted density ($\rho$).

### 1.2. Geometric Kernel (`AlphaHullWrapper`)

- **`compute_hull(points: np.ndarray, alpha: float) -> Trimesh`**
  - Input: 3D point cloud and concavity parameter.
  - Output: Mesh object with `.volume` property.
- **`find_golden_alpha(points: np.ndarray, target: float) -> float`**
  - Input: Calibration point cloud and target unit volume ($V_{\mu}$).
  - Output: The "Golden Alpha" ($\alpha_{golden}$) where $Vol \approx V_{\mu}$.

### 1.3. Physical Safety Utility (`MathUtils.SanityGuard`)

- **`validate_volumetric_bounds(n_vol: float) -> Tuple[bool, str]`**
  - Input: Calculated volumetric count.
  - Output: `(is_valid, reason)` – true if within physically possible range [0, 1000].
- **`check_manifold_safety(points: np.ndarray) -> bool`**
  - Input: Raw point cloud.
  - Output: true if point set is valid for AlphaHull (non-coplanar, N>=4).

### 1.4. Logic Gate (`LogicGate`)

- **`evaluate(perception: PerceptionState) -> GateDecision`**
  - Input: Holistic state containing visible vs volumetric vs spike metrics.
  - Output: `action` (exit/loop) and `anomaly_type`.

## 2. State Contract (RecursiveFlowState)

The LangGraph state is governed by the following Pydantic schemas in `graph_state.py`:

- **`GlobalContext`**: `session_id`, `main_intent`, `unit_volume_prior`, `density_prior`.
- **`PerceptionState`**: `rho`, `golden_alpha`, `total_observed_volume`, `n_volumetric_range`, `fusion_confidence`, `shield_scores`.
- **`DecisionState`**: `status` (loop/exit), `anomaly_type`, `loop_count`.
