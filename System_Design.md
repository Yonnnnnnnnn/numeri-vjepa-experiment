"""
System Design

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol : SystemDesign (this document)

Non-Terminals :
┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
│ <Perception> → v2e, SAM2, CountVid, DINOv2 │
│ <Brain> → V-JEPA, Intent Genesis (VLM+GroundingDINO) │
│ <Controller> → VL-JEPA, CountVid, SAM2, Fusion │
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
- **Depth Anything V2**: Physical truth source for volumetric scaling (Cluster-based estimation).
- **DINOv2**: Semantic feature source for ReID and Density Sensing.

### 1.2. Cognitive Layer (Otak)

- **V-JEPA**: Spatio-temporal world model providing latent memory.
- **Intent Genesis (Step 0)**: A "Scout & Analyst" module that uses GroundingDINO and VLM to generating specific intents and reference crops before the main loop logic begins.
- **Density Engine**: MLP that fuses DINOv2 features with visual specularity to predict packing density ($\rho$).
- **Geometric Kernel**: `GoldenAlphaCalibrator` (Binary Search) for finding the optimal $\alpha$ where $V_{concave} \approx V_{unit}$.

### 1.3. Logical Orchestrator (Pikiran)

- **LangGraph (Sovereignty Chain)**: Implements the 14-step recursive logic (starting from Step 0).
- **Multi-Shield Fusion**: Triangulates Spatial (Spikes), Volumetric (Math), and Latent (JEPA) evidence.
- **Immunity System (Step 12.1/12.2)**: A subsystem within the Director and Fusion Engine that classifies distractors as "Contra Intents". Enhanced by a **Semantic Guard** in V3.3.1 that prevents misclassification of user-requested SKUs (e.g., 'balls') into contra categories and suppresses noise like 'human hands'.
- **Logic Gate**: High-speed deterministic guard for anomaly classification.
- **SLM Reasoner**: Large language logic for intent refinement and discovery.

## 2. Sovereignty Implementation

- **Recursive Intent**: The system treats $N_{vol}$ as the "Physical Truth" and $N_{vis}$ as a "Hypothesis". Discrepancies trigger a re-observation loop.
- **Per-Cluster Weighing (V3.3.3)**: Instead of a global stack volume division, the system map each 3D cluster to its visual label and divides by that specific SKU's prior volume. This allows accurate counting in heterogeneous scenes (e.g., balls mixed with cups).
- **Sovereignty Protocol**: Consensus is reached only when the `FusionConfidence` exceeds the configured threshold across all shields.

## 3. Physical Safety & Performance

### 3.1. Engine Singletons & Model Caching

- To mitigate VRAM fragmentation, heavy models (JEPA, SAM2, DINOv2) are initialized once via a lazy-loading Singleton pattern in `recursive_flow.py`.

### 3.2. Volumetric Guardrails & State Integrity

- **SanityGuard**: Secara proaktif memvalidasi batas volumetrik dan keamanan manifold (point cloud integrity). V3.3.1 menyertakan **Strict Numeric Parser** (Scientific Notation) dan **Safety Floors** untuk nilai V_unit.
- **Parallel State Integrity (Fresh DTO Pattern)**: To prevent "Lost Update" bugs during parallel sensor execution (CountVid, SAM2, V2E), LangGraph nodes return fresh `PerceptionState` DTOs containing ONLY their specific updates. This ensures the LangGraph reducer correctly merges all sensor evidence without overwriting sibling data.
- **Numerical Stability**: V3.3.3 hotfix ensures dynamic VLM parameters (e.g., `max_new_tokens`) are correctly propagated through the `VLMInferenceModel` interface to prevent kernel crashes during high-load reasoning.

### 3.3. Coordinate Scaling

Pemetaan spasial disinkronkan antara loop logika (224x224) dan visualizer (resolusi asli video) melalui transformasi koordinat linier untuk memastikan representasi anomali yang akurat.

### 3.4. Persistent Memory Persistence (Process Sharing)

- To support separate visualization or analysis processes without re-encoding, V-JEPA latent states are persisted to shared disk (`vjepa_context_dump.pt`). This enables "Process-Sharing Fallback" where a decoupled visualizer can recover target object embeddings even if the main orchestration engine is running in a different memory space.

### 3.5. SLM Physical Priors

- **Knowledge-Driven Calibration**: The system utilizes the **SLM Reasoner** not just for anomaly explanation, but as a source of physical world knowledge. It queries the SLM for object volume priors ($V_{unit}$) to bootstrap the volumetric calculation before empirical calibration is achieved.
