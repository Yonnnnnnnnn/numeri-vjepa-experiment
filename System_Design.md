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
- **Intent Genesis (Step 0)**: A "Scout & Analyst" module that uses GroundingDINO and VLM to generating specific intents and reference crops before the main loop logic begins. **V3.5 Global Pre-Scan**: The orchestrator now reads the entire video and samples frames for multi-frame analysis. **V3.6 Discovery Intelligence**: Genesis now extracts **Latent Anchors** — high-fidelity V-JEPA latent vectors — for each discovered brand. It also implements a **Semantic Filter** that prohibit metadata. **V3.7 Multi-Frame Discovery Overhaul**: Implements **Accumulative Visual Fingerprinting** (Scout top 3 frames) and **Prompt-Aware Target Injection** (Biasing VLM labels based on user-provided brand keywords) to resolve identification blindness.
- **Density Engine**: MLP that fuses DINOv2 features with visual specularity to predict packing density ($\rho$).
- **Geometric Kernel**: `GoldenAlphaCalibrator` (Binary Search) for finding the optimal $\alpha$ where $V_{concave} \approx V_{unit}$.

### 1.3. Logical Orchestrator (Pikiran)

- **LangGraph (Sovereignty Chain)**: Implements the 14-step recursive logic (starting from Step 0).
- **Multi-Shield Fusion**: Triangulates Spatial (Spikes), Volumetric (Math), and Latent (JEPA) evidence.
- **Immunity System (Step 12.1/12.2)**: A subsystem within the Director and Fusion Engine that classifies distractors as "Contra Intents". Enhanced by a **Semantic Guard** in V3.3.1 that prevents misclassification of user-requested SKUs (e.g., 'balls') into contra categories and suppresses noise like 'human hands'.
- **Logic Gate**: High-speed deterministic guard for anomaly classification.
- **SLM Reasoner**: Large language logic for intent refinement and discovery.
- **Analyst Override (V3.4)**: When the SLM's object count disagrees with the Scout's visual count during a `volumetric` anomaly loop, the system trusts the SLM (Analyst) and overrides `n_visible`. This breaks the logical deadlock where Scout hallucinates (e.g., 5) but SLM correctly identifies (e.g., 3).

## 2. Sovereignty Implementation

- **Recursive Intent**: The system treats $N_{vol}$ as the "Physical Truth" and $N_{vis}$ as a "Hypothesis". Discrepancies trigger a re-observation loop.
- **Per-Cluster Weighing (V3.3.3)**: Instead of a global stack volume division, the system map each 3D cluster to its visual label and divides by that specific SKU's prior volume. This allows accurate counting in heterogeneous scenes (e.g., balls mixed with cups).
- **Spatial Memory / Residual Intent Mapping (V3.4)**: When a 3D cluster has no visual detection match (e.g., an occluded ball under a cup), the Math Kernel checks which genesis intents are "unaccounted for" and assigns the orphaned volume to the first unaccounted intent. This gives the system "object permanence" — it remembers objects even when they cannot be seen.
- **Sovereignty Protocol**: Consensus is reached only when the `FusionConfidence` exceeds the configured threshold across all shields.

## 3. Physical Safety & Performance

### 3.1. Engine Singletons & Model Caching

- To mitigate VRAM fragmentation, heavy models (JEPA, SAM2, DINOv2) are initialized once via a lazy-loading Singleton pattern in `recursive_flow.py`.

### 3.2. Volumetric Guardrails & State Integrity

- **SanityGuard**: Secara proaktif memvalidasi batas volumetrik dan keamanan manifold (point cloud integrity). V3.3.1 menyertakan **Strict Numeric Parser** (Scientific Notation) dan **Safety Floors** untuk nilai V_unit.
- **Parallel State Integrity (Partial Updates Pattern)**: To prevent "Lost Update" bugs during parallel sensor execution (CountVid, SAM2, V2E), LangGraph nodes return dictionary patches containing ONLY their specific updates. The `merge_perception` reducer ensures these patches are merged without triggering Pydantic default values for unchanged fields.
- **Conditional Genesis Lifecycle**: Step 0 (Intent Genesis) is executing via a conditional router. It runs ONLY once at the start of a session or when `is_genesis_complete` is False, preventing redundant VLM calls and log pollution.
- **Numerical Stability**: V3.3.3 hotfix ensures dynamic VLM parameters (e.g., `max_new_tokens`) are correctly propagated through the `VLMInferenceModel` interface to prevent kernel crashes during high-load reasoning.
- **BGR→RGB Color Correction (V3.4)**: Frames from OpenCV (`cv2.VideoCapture`) are now converted from BGR to RGB before being passed to VLM/PIL. Without this, Red↔Blue channels are swapped, causing VLM color hallucinations (e.g., "Blue Cup" instead of "Red Cup").
- **Anti-Stuttering VLM Parameters (V3.4)**: `VLMInferenceModel.predict()` now includes `repetition_penalty=1.15`, `temperature=0.7`, and `do_sample=True` by default. This prevents the VLM from getting stuck in repetition loops (e.g., outputting "INTENT: Blue Plastic Cup" 9 times).

### 3.6. Circuit Breaker & Resilience

- **Stale Data Guard**: Before Fusion, the system validates the freshness of volumetric data. If data is detected as "Stale" (out of sync with current frame), the **Circuit Breaker** (Logic Gate) triggers a "Degraded Status" exit instead of a reasoning loop to prevent logical deadlock.

### 3.3. Coordinate Scaling

Pemetaan spasial disinkronkan antara loop logika (224x224) dan visualizer (resolusi asli video) melalui transformasi koordinat linier untuk memastikan representasi anomali yang akurat.

### 3.4. Persistent Memory Persistence (Process Sharing)

- To support separate visualization or analysis processes without re-encoding, V-JEPA latent states are persisted to shared disk (`vjepa_context_dump.pt`). This enables "Process-Sharing Fallback" where a decoupled visualizer can recover target object embeddings even if the main orchestration engine is running in a different memory space.

### 3.5. SLM Physical Priors

- **Knowledge-Driven Calibration**: The system utilizes the **SLM Reasoner** not just for anomaly explanation, but as a source of physical world knowledge. It queries the SLM for object volume priors ($V_{unit}$) to bootstrap the volumetric calculation before empirical calibration is achieved.
