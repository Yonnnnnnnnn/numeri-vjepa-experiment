# Recursive Intent Implementation Plan: Technical Detail (Hybrid Logic V2)

## 1. Overview

**Goal**: Implement adaptive multi-object counting using **Recursive Intent** with a focus on **Real-Time Performance** and **Hallucination Resistance**.

**Core Strategy**:

1.  **Hybrid Decision Architecture**: Replace the slow "SLM Council" with a fast **Logic Gate (Math Guards)** that handles 90% of frames. SLMs ("Targeted SLM Judge") are only triggered for ambiguous cases.
2.  **3D Point Cloud Projection (SAM2/Depth V2)**: Use Monocular Depth Estimation (**Depth Anything V2**) to calculate relative volume, avoiding the computational cost and instability of full 3D reconstruction.
3.  **LangGraph Orchestration**: Manage the stateful loop between Detection, Logic Check, and Refinement using a **Scoped State** architecture.

## 2. Component Roles & Interfaces

| Component        | Role                          | Logic Type                 | Use Case                                                                                 |
| ---------------- | ----------------------------- | -------------------------- | ---------------------------------------------------------------------------------------- |
| **LangGraph**    | Workflow Orchestrator         | State Machine              | Routing traffic between Fast Path (Exit) and Slow Path (Loop).                           |
| **Logic Gate**   | Primary Decision Maker        | Rule-Based (Deterministic) | High confidence checks, BBox overlap validation. Speed: <10ms.                           |
| **Targeted SLM** | Ambiguity Resolver            | Probabilistic (LLM)        | "Is this blob a cup or shadow?" triggered only when Logic Gate fails.                    |
| **V2E**          | Event-Based Sensor (Parallel) | Physics Simulation         | Generates high-sensitivity event spikes from standard video for anomaly detection.       |
| **FusionEngine** | Anomaly Detector              | Hybrid Logic               | Mendeteksi **Residu Spike** dengan **Motion Compensation** (filter jitter kamera).       |
| **MathUtils**    | Mathematical Kernel           | Pure Math                  | Mendapatkan $N_{volumetric}$ menggunakan **Unit Reference** (volume per kategori).       |
| **SafetyGuard**  | Identity & Integrity          | Rule-Based (Deterministic) | Mencegah double counting dan memastikan konsistensi identitas objek antar frame.         |
| **VL-JEPA**      | Director (RGB)                | Vision-Language            | Identifying "Reference Object" at $t=0$. Feeds intent to SLM.                            |
| **DINOv2 + MLP** | Occupancy Detector            | Semantic Textures          | `sklearn.MLPRegressor` predicts $\rho$ from DINOv2 texture tokens.                       |
| **V-JEPA**       | Spatio-Temporal Memory        | Self-Supervised Learning   | Uses **Context Tokens (`ctxt`)** from `jepa-main` to maintain persistent 3D world model. |
| **AlphaShape**   | Hull Wrapper                  | Computational Geometry     | Library: `alphashape`. Wraps point clouds with **Golden Alpha** tightness.               |
| **DepthEngine**  | 3D Perception                 | Monocular Depth            | Extracting metric depth maps calibrated by the reference object.                         |

### 2.1. Note on Bayesian Consistency

While we use **Hybrid Logic** for speed, the system conceptually maintains **Bayesian Integrity**:

- **Prior**: V-JEPA's temporal memory.
- **Likelihood**: Logic Gate's confidence checks.
- **Posterior**: The accumulation of confidence through recursive loops.
- **Evidence**: The output from SAM2+DepthAnything (3D) Estimator.

This ensures that while the implementation is fast (Heuristic), the logic remains mathematically grounded (Bayesian). `MathUtils` acts as the shared engine for these operations.

## 3. System Architecture (Hybrid Flow)

```mermaid
flowchart TD
    subgraph Sensors[Parallel Input Stream]
        direction TB
        RAW[Video Stream]
        RAW -->|Standard RGB| VJEPA[V-JEPA Brain]
        RAW -->|Standard RGB| VLJEPA[VL-JEPA Director]
        RAW -->|Standard RGB| COUNT[SAM2+DepthAnything (3D)]
        RAW -->|Frame Conv| V2E[V2E Spike Sensor]
    end

    subgraph Fast_Path[Fast Path: Every Frame]
        VJEPA -->|Memory| FUSION[Fusion Engine]
        COUNT_GD[CountGD] -->|N_visible| FUSION
        SAM2[SAM2+Depth] -->|Point Cloud| FUSION
        V2E -->|Spikes| FUSION
        FUSION --> LOGIC{Logic Gate}
    end

    subgraph Slow_Path[Slow Path: On Ambiguity]
        LOGIC -->|Unexplained Spikes| SLM[Targeted SLM Judge]
        SLM -->|New Context| VLJEPA
        VLJEPA -->|Refined Intent| COUNT
    end

    subgraph Exit
        LOGIC -->|Confident| FINAL[Final Output]
        SLM -->|Reject/Confirm| FINAL
    end

    classDef fast fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef slow fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    class LOGIC,COUNT,V2E,VJEPA,FUSION fast;
    class SLM,VLJEPA slow;
```

## 4. State Estimation Framework (3D Point Cloud Projection & Scoped State)

### 4.1. Scoped State Architecture (Solving the "God Object")

To ensure debuggability and prevent race conditions, we split the global state into 4 isolated Pydantic models:

```python
class GlobalContext(BaseModel):
    session_id: str
    main_intent: List[str]  # e.g., ["Gelas"]
    start_time: float

class PerceptionState(BaseModel):
    """Owned by V2E, SAM2, DepthEstimator"""
    current_frame_idx: int
    raw_detections: List[Dict]
    depth_map_stats: Dict  # {mean_depth: 0.5, has_depth: True}
    unexplained_blobs: List[Dict]

class DecisionState(BaseModel):
    """Owned by LogicGate and TargetedSLM"""
    status: Literal["processing", "looping", "exit"]
    logic_gate_result: Dict  # {rule_applied: "Rule 1", confidence: 0.9}
    slm_reasoning: Optional[str]
    loop_count: int = 0

class RecursiveFlowState(TypedDict):
    """The root container for LangGraph"""
    ctx: GlobalContext
    perception: PerceptionState
    decision: DecisionState
    output: List[Dict]
```

### 4.2. Logic Gate Rules (The "Math Guards")

Instead of voting, we use hard thresholds to filter noise:

1.  **Rule 1 (Pass)**:
    - `Confidence > 0.85` AND `Unexplained_Blob_Area < 10%`
    - **Action**: EXIT (Accept Count).
2.  **Rule 2 (Fail/Reject)**:
    - `Confidence < 0.4`
    - **Action**: IGNORE (Noise).
3.  **Rule 3 (Ambiguous - Trigger SLM)**:
    - `Confidence` between 0.4 - 0.85 OR `Unexplained_Blob_Area > Threshold`
    - **Action**: LOOP (Wake up SLM).

### 4.3. 3D Volumetric Counting (Hybrid Geometric-Semantic Architecture)

The system uses a **Decomposed 3D Counting (3DC)** approach, separating volume estimation from density prediction:

1.  **Reference-First Calibration (The Anchor)**:
    - At $t=0$, the system identifies a single "Template Object" (e.g., one cup).
    - SLM provides the object's physical dimensions ($V_{\mu}$).
    - **Metric Scale**: Pixels are mapped to CM based on the known height.
    - **Alpha Calibration**: Uses `alphashape.optimize()` logic to find the **"Golden Alpha"** value where $V_{calc} \approx V_{\mu}$.
2.  **Geometry Estimation (V-JEPA Memory + AlphaShape)**:
    - SAM2 segments objects, and DepthAnything provides metric depth maps (now calibrated).
    - V-JEPA's **Persistent Latent Context** (using `ctxt` tokens) acts as a pose proxy, fusing point clouds from a 180° orbit.
    - **Alpha-Concave Hull** (via `alphashape` library) wraps the accumulated point cloud using the locked Golden Alpha.
3.  **Density Prediction (DINOv2 + sklearn MLP)**:
    - The **DINOv2 Engine** analyzes RGB (specular highlights) + Depth texture.
    - A `sklearn.MLPRegressor` (trained or heuristic-initialized) maps texture tokens to **Occupancy Ratio ($\rho$)**.
4.  **Final Volumetric Count ($N$)**:
    $$N = \frac{V_{\text{stack}} \times \rho}{V_{\mu}}$$

---

## 5. Detailed Interaction Flow

### Step 0: Reference Calibration (Phase 0)

- **Input**: First frame of the video.
- **Identification**: VL-JEPA + SLM anchor the metric context (Scale, $V_{\mu}$, and Golden Alpha).
- **Effect**: Locks the coordinate system and geometric parameters before counting starts.

### Step 1: Perception & Orbit (Phase 1)

- **Accumulation**: Camera orbits the stack (180°). V-JEPA builds the spatio-temporal memory.
- **Perception**: CountGD tracks $N_{visible}$, SAM2/Depth provides point cloud frames.
- **Fusion**: Point clouds are registered into the persistent V-JEPA context.

### Step 2: Volumetric Audit & Occupancy (Phase 2)

- **Calculation**: `MathUtils` computes $V_{\text{stack}}$ via `alphashape` hull.
- **Occupancy**: DINOv2 + `sklearn.MLPRegressor` predicts $\rho$.
- **Conflict Detection**: Logic Gate compares $N_{volumetric}$ vs. $N_{visible}$ and checks for **Residual Spikes**.

### Step 3: Targeted SLM $\to$ VL-JEPA (Sutradara)

1.  **Reasoning**: SLM analyzes discrepancy between visual, volumetric, and event sensors.
2.  **Morphism**: SLM adjusts $\rho$, Scale, or Alpha parameters if discrepancy persists.
3.  **Refinement**: System reruns the loop with updated intentions or parameters.

## 4. Taxonomy of Recursive Loops

Sistem membedakan dua skenario loop berdasarkan sumber "kejutan" (_Surprise Signal_):

| Loop Type           | Trigger (Surprise)                    | Object Input       | Goal (Action)            | Bayesian Effect                                    |
| :------------------ | :------------------------------------ | :----------------- | :----------------------- | :------------------------------------------------- |
| **Discovery Loop**  | High Residual Spike di luar area Mask | Null (Area Kosong) | Membuat **Intent Baru**  | Menambah entitas baru di ruang probabilitas        |
| **Refinement Loop** | Volume Discrepancy di dalam area Mask | Objek Terlacak     | Menguji **Oklusi/Lapis** | Meningkatkan keyakinan (Updating Count/Confidence) |

> [!IMPORTANT]
> **Concurrency Note**: Kedua loop ini berjalan **secara paralel** dalam tahap persepsi. Jika sebuah frame memiliki kedua anomali, Logic Gate akan membundel keduanya ke dalam satu instruksi SLM tunggal. Sistem hanya melakukan **SATU loop rekursif** untuk menyelesaikan semua masalah di frame tersebut secara simultan.

## 6. Implementation Phases

### Phase 0: Setup & Graph Definition

- Setup SAM2 (Executor) & Depth Anything V2.
- Implementasikan `MathUtils.back_project()` untuk menghasilkan Point Cloud.

### Phase 1: Perceptual Feedback (Spatial Mismatch)

- [ ] Implementasikan **Residual Spike Calculation**: Hitung energi spike di luar area BBox RGB.
- [ ] Tambahkan deteksi "Unidentified Visual Patches" berdasarkan residu tersebut.

### Phase 2: Hybrid Decision Gate (Identity Guarded)

- [ ] Implementasikan Logic Gate yang mengecek **Depth Protrusion** pada area anomali.
- [ ] **Identity Guard Implementation**: Tambahkan pengecekan **Spatial & Vector Similarity** sebelum memicu loop atau mengupdate count untuk mencegah double counting (Problem 6).
- [ ] Tambahkan **Targeted SLM Node**: Hanya dipicu jika (Residu Spike > T) AND (Depth > T) AND (Not a redundant identity).

### Phase 3: Recursive Re-Identification

- Update `VLJEPA` to accept explicit feedback from SLM.

### Phase 4: Relative Depth Integration

- Integrate `Depth Anything V2`.
- Implement `estimate_volume_heuristic` function.
- Validate "Stacking" logic using depth sums.

## 7. Key Code Structure

### Logic Gate Node (Efficient)

```python
def logic_gate_node(state: GraphState):
    # 1. Math Guard
    if state.min_confidence > 0.85 and state.unexplained_area < 0.1:
        return {"decision": "exit"}

    # 2. Ambiguity Trigger
    if state.unexplained_area > 0.3:
        return {
            "decision": "loop",
            "trigger_reason": "large_blob",
            "blob_location": state.largest_blob_bbox
        }

    return {"decision": "exit"} # Default safe exit
```

### Targeted SLM Node (Reasoning)

```python
def targeted_slm_node(state: GraphState):
    # Passes Mathematical Evidence to SLM for Reasoning
    prompt = f"""
    [PHYSICAL EVIDENCE]
    - Visible Count (SAM2): {state.count_visible}
    - Volumetric Prediction (MathUtils): {state.count_volumetric}
    - Residual Spike Energy: {state.spikes_residue}

    [TASK]
    Reason if the volumetric data justifies a hidden object count.
    If YES, hypothesize the hidden object type and return instruction for VL-JEPA.
    """
    response = llm.invoke(prompt)
    return {"instruction_to_vljepa": response}
```

## 8. Conclusion

By removing the "SLM Council" and adopting a **Hybrid** approach, we reduce per-frame latency from ~5000ms (3 LLMs voting) to ~100ms (Logic Gate) + occasional 500ms (Targeted SLM).
By using **3D Point Cloud** via **Depth Anything V2**, we gain mathematically valid volume estimation without the implementation risks of full 3D reconstruction.
