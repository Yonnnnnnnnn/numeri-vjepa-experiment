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

## 1. Engine Interfaces

### 1.1. V-JEPA Engine (`VJEPAEngine`)

- **`encode(frame_tensor: Tensor) -> Latent`**
  - Input: Normalized RGB frame or event-reconstructed frame (B, 3, 224, 224).
  - Output: Latent representation (B, N, 1024).
- **`predict_next_state(steps: int) -> LatentPrediction`**
  - Purpose: Occlusion reasoning via temporal prediction.

### 1.2. Director (`VLJEPAEngine`)

- **`generate_intent(latent: Latent, prompt: str) -> Intent`**
  - Input: V-JEPA latent features and a high-level task description.
  - Output: Structured instruction for the Executor (e.g., "Count Milk cartons").

### 1.3. Executor (`CountGDEngine`)

- **`count_objects(frame: Tensor, intent: Intent) -> Results`**
  - Input: Raw frame and structured visual prompt.
  - Output: Bounding boxes, confidence scores, and instance counts.

## 2. Feedback Loops (Recursive Intent)

- **`update_intent(failure_signal: Reflection) -> NewIntent`**
  - Triggered when the Executor detects an anomaly or low-confidence match.
