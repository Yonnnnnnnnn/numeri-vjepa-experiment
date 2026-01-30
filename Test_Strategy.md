# Test Strategy: Vision & Loop Verification

This document outlines the strategy for verifying the Antigravity V2 pipeline.

## 1. Unit Testing

- **VJEPAEngine**: Verify weight loading, encoding shape, and prediction consistency.
- **Preprocessing**: Test spike-to-frame conversion and normalization.

## 2. Integration Testing

- **Recursive Flow**: Verify end-to-end execution from raw video to final count using `run_recursive_system.py`.
- **Hybrid Decision Gate**: Validate that Logic Gate correctly routes frames based on spatial (V2E) and volumetric (SAM2+Depth) anomalies.
- **V-JEPA Memory**: Verify that temporal latents provide consistent context for the countgd_executor_node.

## 3. Performance Metrics

- **Accuracy**: Counting error percentage in high-occlusion scenarios.
- **Latency**: End-to-end inference time per frame step (Target <150ms on T4/L4).
- **Volumetric Precision**: Discrepancy between $N_{visible}$ and $N_{volumetric}$ in stacked scenarios.
