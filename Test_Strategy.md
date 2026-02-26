# Test Strategy: Sovereignty & Physical Truth

This document outlines the strategy for verifying the Antigravity V3.1 "Sovereignty" protocol.

## 1. Unit Testing (Kernels)

- **VJEPAEngine**: Verify `PersistentLatentContext` buffer mechanics and latent-pose stability.
- **Geometric Kernel**: Test `AlphaHullWrapper` for volume accuracy and Golden Alpha binary search convergence.
- **Density Engine**: Validate MLP Regressor prediction bounds and specularity analysis robustness.
- **MathUtils**: Verify `calculate_volumetric_count` against known stack sizes.

## 2. Integration Testing (Sovereignty Chain)

- **Recursive LangGraph**: Verify the 13-step flow using `test_phase5_orchestration.py`.
- **Multi-Shield Fusion**: Validate that weighted consensus correctly handles single-shield failures (e.g., occlusion blocking SAM2 but not V2E).
- **Discovery Loop**: Test the SLM's ability to inject new labels into the `main_intent` during volumetric anomalies.

## 3. Runtime Debugging & Stability

- **Singleton Enforcement**: Verify that heavy engines (V-JEPA, SAM2) are cached as singletons to prevent VRAM overflow.
- **Edge Case Sanity**: Test `SanityGuard` against division-by-zero or empty point clouds.

- **Graceful Degradation**: Ensure the graph reaches `exit` even if secondary sensors (e.g., v2e) return empty data.
- **Per-Cluster Volume Validation (V3.8.1)**: Verify that `c_vol` (per-cluster meter cubes) is correctly extracted from the depth manifold across multiple isolated stacks. Failure condition: $n_{vol}$ exceeding physically plausible range due to global volume sum divide.
- **Offline Protocol Verification**: Disable internet access and verify system initializes sub-10s using local HuggingFace cache. Verify `local_files_only` fallback logic.
- **Density Cold-Start Validation**: Check that `DensityPredictor` returns $\rho \in [0.1, 20.0]$ upon first use before any `fit()` calls from external data.

## 4. VLM & SLM Component Verification (V3.3.3)

- **VLM Interface Integrity**: Verify that `VLMInferenceModel` correctly handles dynamic keyword arguments (`max_new_tokens`) via `tests/test_vlm_interface.py`.
- **Semantic Pivot Logic**: Validate the Generic Anchor Strategy (Specific label -> Basic noun) using `Implementation/tests/test_semantic_pivot_fast.py`.
- **(V3.8) Defensive Input Processing**:
  - Verify tensor batching stability by feeding variant resolution frames into `VJEPAEngine.encode()`.
  - Validate coordinate parsing logic for GroundingDINO list-format boxes.
  - Smoke test the Semantic Filter against instruction-heavy prompts to ensure zero "Label Pollution".
- **(V4.0) Latent Locking & Vector Mathematics**:
  - Verify `test_latent_locking.py` confirms that V-JEPA object extraction maintains identical cosine similarity vectors for strict threshold arbitration ($>0.85$).
