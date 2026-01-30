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

- **v2e**: Converts video into high-temporal spike events for anomaly detection.
- **SAM2**: Performs high-fidelity instance segmentation.
- **Depth-Anything V2**: Generates relative depth maps for 3D point cloud projection.
- **DINOv2**: Extracts semantic features for object re-identification.

### 1.2. World Model (Brain)

- **V-JEPA**: A Joint-Embedding Predictive Architecture. It maintains spatio-temporal memory and predicts future states.

### 1.3. Logical Controller (Pikiran)

- **VL-JEPA (Director)**: Vision-language director that sets and updates the target "Intent".
- **CountGD (Visual Executor)**: Provides zero-shot visual counting ($N_{visible}$).
- **Logic Gate (Math Guard)**: Fast rule-based decision gate for anomaly detection (Spatial/Volumetric).
- **Targeted SLM (Judge)**: Ambiguity resolver triggered by Logic Gate for deep reasoning.

## 2. Strange Loop Implementation

- **Recursive Intent**: The system self-refines its task list based on counting discrepancies.
- **Volumetric Counting**: Combines Depth and SAM2 to estimate counts via physical volume.
