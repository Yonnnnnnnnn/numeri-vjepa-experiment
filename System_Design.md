"""
System Design

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol : SystemDesign (this document)

Non-Terminals :
┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
│ <Perception> → v2e, SAM2, CountVid, DINOv2 │
│ <Brain> → V-JEPA │
│ <Controller> → VL-JEPA, CountVid, SAM2 │
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

- **v2e**: Asynchronous event source for residue detection.
- **SAM2**: Union Masking and Cluster Detection for occlusion management.
- **Depth Anything V2**: Physical truth source for volumetric scaling.
- **DINOv2**: Semantic feature source for ReID and Density Sensing.

### 1.2. Cognitive Layer (Otak)

- **V-JEPA**: Spatio-temporal world model providing latent memory.
- **Density Engine**: MLP that fuses DINOv2 features with visual specularity to predict packing density ($\rho$).
- **Geometric Kernel**: `alphashape` binary search for Golden Alpha calibration.

### 1.3. Logical Orchestrator (Pikiran)

- **LangGraph (Sovereignty Chain)**: Implements the 13-step recursive logic.
- **Multi-Shield Fusion**: Triangulates Spatial (Spikes), Volumetric (Math), and Latent (JEPA) evidence.
- **Logic Gate**: High-speed deterministic guard for anomaly classification.
- **SLM Reasoner**: Large language logic for intent refinement and discovery.

## 2. Sovereignty Implementation

- **Recursive Intent**: The system treats $N_{vol}$ as the "Physical Truth" and $N_{vis}$ as a "Hypothesis". Discrepancies trigger a re-observation loop.
- **Sovereignty Protocol**: Consensus is reached only when the `FusionConfidence` exceeds the configured threshold across all shields.

## 3. Physical Safety & Performance

### 3.1. Engine Singletons & Model Caching

- To mitigate VRAM fragmentation, heavy models (JEPA, SAM2, DINOv2) are initialized once via a lazy-loading Singleton pattern in `recursive_flow.py`.

### 3.2. Volumetric Guardrails (SanityGuard)

- **SanityGuard**: Secara proaktif memvalidasi batas volumetrik dan keamanan manifold (point cloud integrity) untuk mencegah degradasi performa atau _crash_ pada kernel geometris.
- Numerical stability triggers on division-by-zero or non-manifold mesh generation in the AlphaHull kernel.

### 3.3. Coordinate Scaling

Pemetaan spasial disinkronkan antara loop logika (224x224) dan visualizer (resolusi asli video) melalui transformasi koordinat linier untuk memastikan representasi anomali yang akurat.
