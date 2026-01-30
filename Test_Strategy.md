# Test Strategy: Vision & Loop Verification

This document outlines the strategy for verifying the Antigravity V2 pipeline.

## 1. Unit Testing

- **VJEPAEngine**: Verify weight loading, encoding shape, and prediction consistency.
- **Preprocessing**: Test spike-to-frame conversion and normalization.

## 2. Integration Testing

- **Inference Loop**: Verify that SAM2 masks and Depth maps are correctly used for volumetric counting.
- **Recursive Intent**: Use the `run_recursive_system.py` runner to verify end-to-end flow on sample video data.
- **Logic Gate**: Unit test the anomaly detection rules (Spatial vs Volumetric).

## 3. Performance Metrics

- **Accuracy**: Counting error percentage in high-occlusion scenarios.
- **Latency**: End-to-end inference time per frame step (Target <100ms on L40S).
- **Permanence**: Time duration an object can be "remembered" while occluded.
