"""
System Design

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol : SystemDesign (this document)

Non-Terminals :
┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
│ <Perception> → v2e, SAM2, CountGD, DINOv2 │
│ <Brain> → V-JEPA │
│ <Controller> → VL-JEPA, CountGD, SAM2 │
└───────────────────────────────────────────────────────────────────────────┘

Terminals : str, component_names

Production Rules:
SystemDesign → <Perception> <Brain> <Controller>
═══════════════════════════════════════════════════════════════════════════════
"""

# System Design: V-JEPA Event-Based Intelligence

The Antigravity V2 system is a high-speed inventory counting and auditing platform that utilizes asynchronous spike events and temporal predictive models.

## 1. Core Components

### 1.1. Perception Pipeline (Mata)

- **v2e**: Converts video/sim into biological-like spike events.
- **SAM2**: Performs high-fidelity segmentation on event-reconstructed frames.
- **Depth Anything V2**: Provides monocular depth estimation for 3D volumetric analysis.
- **DINOv2**: Extracts semantic features from masks for object re-identification.

### 1.2. World Model (Brain)

- **V-JEPA**: A Joint-Embedding Predictive Architecture trained on video. It predicts future latents, enabling the system to "remember" objects during occlusions.

### 1.3. Logical Controller (Pikiran)

- **VL-JEPA (Director)**: A vision-language model that sets goals and interprets scene context.
- **CountGD (Visual Executor)**: Provides zero-shot object counting and bounding boxes.
- **Logic Gate (Math Guards)**: Deterministic rule-based filter that handles anomaly detection and routing.
- **Targeted SLM (Reasoning)**: Triggered for high-level ambiguity resolution and intent refinement.

## 2. Strange Loop Implementation

- **Recursive Intent**: The system self-corrects its "Intent" based on counting anomalies.
- **Sensory-Predictive Loop**: Expectations from the Brain modify the sensitivity of the Eyes (v2e).
