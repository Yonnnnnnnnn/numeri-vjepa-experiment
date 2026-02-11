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
