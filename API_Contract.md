"""
API Contract

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol : APIContract (this document)

Non-Terminals :
┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
│ <EngineAPI> → VJEPAEngine | SLMEngine (Unified VLM) | Director | Executor │
│ <LoopAPI> → Feedback mechanisms │
│ <LegacyAPI> → VLJEPAEngine (PaliGemma) - Deprecated in V4.0 │
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
- **`generate_saliency_map(image: np.ndarray) -> np.ndarray`**
  - Input: RGB image.
  - Output: (16x16) normalized saliency heatmap.
- **`find_hotspot_roi(saliency_map) -> List[Tuple]`**
  - Input: Aggregated saliency map.
  - Output: List of (x1, y1, x2, y2) PointBeam ROIs.

### 1.1.5. Intent Genesis (`IntentGenesisModule`)

- **`Step 0: Bio-Inspired Scouting`**
  - Process: Peripheral Saccade (DINOv2) -> PointBeam Fixation -> Foveated Genesis (VLM).
  - Output: `active_intent` (Multi-SKU), `focus_roi` (PointBeam), `video_frames`.
  - Action: Discovers specific brand labels and high-interest ROIs before starting the main loop.

### 1.2. Geometric Kernel (`AlphaHullWrapper` & `GoldenAlphaCalibrator`)

- **`calibrate(clusters, v_unit) -> CalibrationResult`**
  - Input: Point cloud clusters + V_unit target.
  - Output: Optimal `golden_alpha` where $V_{concave} \approx V_{target}$.
- **`compute_hull(points: np.ndarray, alpha: float) -> Trimesh`**
  - Input: 3D point cloud and concavity parameter.
  - Output: Mesh object with `.volume` property.

### 1.3. Physical Safety Utility (`MathUtils.SanityGuard`)

- **`validate_volumetric_bounds(n_vol: float) -> Tuple[bool, str]`**
  - Input: Calculated volumetric count.
  - Output: `(is_valid, reason)` – true if within physically possible range [0, 1000].
- **`check_manifold_safety(points: np.ndarray) -> bool`**
  - Input: Raw point cloud.
  - Output: true if point set is valid for AlphaHull (non-coplanar, N>=4).

### 1.5. Targeted SLM (`SLMEngine`)

- **`generate_initial_intents(prompt, frame, detections) -> List[Dict]`**
  - Input: User prompt, Frame (full or crop) + optional detections.
  - Output: List of specific intent labels vs contra labels with bounding boxes.
  - **V4.0 Unified VLM**: This is now the primary semantic engine for the entire system, replacing legacy PaliGemma proxies.
- **`generate_reasoning(image, anomaly_type, context) -> ReasoningResult`**
  - Input: Current frame, anomaly classification, and state context.
  - Output: Textual explanation and next-step hypothesis.
- **`estimate_object_volume(label: str) -> float`**
  - Input: Object label (e.g., "bottle").
  - Output: SLM-derived physical volume prior ($V_{prior}$) in $m^3$.

### 1.6. V-JEPA Engine (`VJEPAEngine`)

- **`encode(video_tensor) -> Latent`**
  - Input: Temporal frame sequence.
  - Output: Hierarchical latent feature map.
- **`export_context(path: str)`**
  - Input: Target file path.
  - Action: Persists latent state to disk for external process sharing (Visualizer).

## 2. State Contract (RecursiveFlowState)

The LangGraph state is governed by the following Pydantic schemas in `graph_state.py`:

- **`GlobalContext`**: `session_id`, `main_intent`, `user_prompt` (V3.3), `unit_volume_prior`, `density_prior`.
- **`PerceptionState`**: `genesis_intents` (V3.3), `contra_intents` (V3.3), `negative_masks` (V3.3), `rho`, `golden_alpha`, `total_observed_volume`, `n_volumetric_range`, `fusion_confidence`, `shield_scores`.
- **`DecisionState`**: `status` (loop/exit), `anomaly_type`, `loop_count`, `is_genesis_complete` (V3.3.4).
