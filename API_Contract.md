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

### 1.1. Input & Spikes (`V2EEngine`)

- **`generate_events(frame: np.ndarray, timestamp: float) -> np.ndarray`**
  - Input: Raw RGB frame and relative timestamp.
  - Output: Event spike array (x, y, p, t).

### 1.2. Brain (`VJEPAEngine`)

- **`encode(frame_tensor: Tensor) -> Latent`**
  - Input: Normalized RGB frame or event-reconstructed frame (B, 3, 224, 224).
  - Output: Latent representation (B, N, 1024).

### 1.3. Director (`VLJEPAEngine`)

- **`identify_intent(frame: np.ndarray) -> str`**
  - Input: Initial RGB frame.
  - Output: Textual intent (e.g., "cup").
- **`generate_intent(latent: Latent, prompt: str) -> Intent`**
  - Input: V-JEPA latent features and a high-level task description.
  - Output: Structured instruction for the Executor.

### 1.4. Executor (`CountGDEngine` & `SegmentationEngine`)

- **`count(image: np.ndarray, prompt: str) -> (count, detections)`**
  - Input: Raw frame and textual prompt.
  - Output: Integer count and detection metadata.
- **`segment_frame(image: np.ndarray) -> SegmentResult`**
  - Input: Raw frame.
  - Output: List of binary masks and bounding boxes.

## 2. Feedback Loops (Recursive Intent)

- **`update_intent(failure_signal: Reflection) -> NewIntent`**
  - Triggered when the Executor detects an anomaly or low-confidence match.
