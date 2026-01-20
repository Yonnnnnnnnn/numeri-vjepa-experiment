# Test Strategy: Vision & Loop Verification

This document outlines the strategy for verifying the Antigravity V2 pipeline.

## 1. Unit Testing

- **VJEPAEngine**: Verify weight loading, encoding shape, and prediction consistency.
- **Preprocessing**: Test spike-to-frame conversion and normalization.

## 2. Integration Testing

- **Inference Loop**: Verify that SAM2 masks are correctly passed to DINOv2 and then to the DBSCAN clusterer.
- **Recursive Intent**: Use mock anomalies to verify that the Director (VL-JEPA) generates a new instruction.

## 3. Performance Metrics

- **Accuracy**: Counting error percentage in high-occlusion scenarios.
- **Latency**: End-to-end inference time per frame step (Target <100ms on L40S).
- **Permanence**: Time duration an object can be "remembered" while occluded.
