# Recursive Intent Implementation Plan: Technical Detail (Hybrid Logic V2)

## 1. Overview

**Goal**: Implement adaptive multi-object counting using **Recursive Intent** with a focus on **Real-Time Performance** and **Hallucination Resistance**.

**Core Strategy**:

1.  **Hybrid Decision Architecture**: Replace the slow "SLM Council" with a fast **Logic Gate (Math Guards)** that handles 90% of frames. SLMs ("Targeted SLM Judge") are only triggered for ambiguous cases.
2.  **3D Point Cloud Projection (SAM2/Depth V2)**: Use Monocular Depth Estimation (**Depth Anything V2**) to calculate relative volume, avoiding the computational cost and instability of full 3D reconstruction.
3.  **LangGraph Orchestration**: Manage the stateful loop between Detection, Logic Check, and Refinement using a **Scoped State** architecture.

## 2. Component Roles & Interfaces

| Component                | Role                            | Logic Type                 | Use Case                                                                                    |
| ------------------------ | ------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------- |
| **LangGraph**            | Workflow Orchestrator           | State Machine              | Routing traffic between Fast Path (Exit) and Slow Path (Loop).                              |
| **Logic Gate**           | Primary Decision Maker          | Rule-Based (Deterministic) | High confidence checks, BBox overlap validation. Speed: <10ms.                              |
| **Targeted SLM**         | Ambiguity Resolver              | Probabilistic (LLM)        | "Is this blob a cup or shadow?" triggered only when Logic Gate fails.                       |
| **V2E**                  | Event-Based Sensor (Parallel)   | Physics Simulation         | Generates high-sensitivity event spikes from standard video for anomaly detection.          |
| **FusionEngine**         | Anomaly Detector                | Hybrid Logic               | Mendeteksi **Residu Spike** dengan **Motion Compensation** (filter jitter kamera).          |
| **MathUtils**            | Mathematical Kernel             | Pure Math                  | Mendapatkan $N_{volumetric}$ menggunakan **Unit Reference** (volume per kategori).          |
| **SafetyGuard**          | Identity & Integrity            | Rule-Based (Deterministic) | Mencegah double counting dan memastikan konsistensi identitas objek antar frame.            |
| **V-JEPA**               | Spatio-Temporal Memory          | Self-Supervised Learning   | **Spatio-Temporal Anchoring**: Membedakan objek identik berdasarkan koordinat.              |
| **VL-JEPA**              | Director (RGB)                  | Vision-Language            | Generating/Updating Intent List based on feedback from the Slow Path.                       |
| **CountGD**              | Visual Executor                 | Zero-Shot Counting         | Menghitung objek yang terlihat secara langsung ($N_{visible}$).                             |
| **SAM2 + DepthAnything** | 3D Perception Kernel (Executor) | 3D Point Cloud             | Mencari objek (SAM2) dan memproyeksikannya ke ruang 3D (DepthAnything) secara terintegrasi. |

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

### 4.3. 3D Volumetric Counting (SAM2 + DepthAnything)

Sistem menggunakan pendekatan **Point Cloud Back-projection** untuk mendapatkan volume fisik tanpa dataset 3D:

1.  **Segmentation (SAM2)**: Mendapatkan pixel-perfect mask dari objek terdeteksi.
2.  **Depth (DepthAnything)**: Mendapatkan map kedalaman relatif.
3.  **3D Projection (MathUtils)**: Menggunakan formula intrinsik kamera untuk memproyeksikan pixel $(u, v)$ ke koordinat dunia $(x, y, z)$.
    $$z = Depth(u,v)$$
    $$x = (u - c_x) \times z / f_x$$
    $$y = (v - c_y) \times z / f_y$$
4.  **Lattice & Riemann Sums**: Menghitung volume berdasarkan densitas point cloud di dalam hull 3D.
5.  **Heuristic Optimization (CountNet3D inspired)**:
    - **Downsampling**: Reduksi kerapatan point cloud untuk efisiensi real-time ($< 50ms$).
    - **Physical Bounds Validation**: Memastikan $\sum V_{objects} \leq V_{bounding\_box} \times PackingFactor$.
    - **Volume Range Tracking**: Menggunakan nilai [Min, Max] volume per kategori untuk menangani variabilitas produk.

---

## 5. Detailed Interaction Flow

### Step 0: Parallel Acquisition & Initial Intent

- **Stream A (RGB)**: Video frames flow to V-JEPA, VL-JEPA, CountGD, and SAM2.
- **Initial Hint**: V-JEPA processes the **First Frame** to generate a latent representation.
- **Director Wake-up**: VL-JEPA uses this first latent to set the **Initial Intent** (e.g., "Gelas").
- **Stream B (Spikes)**: V2E converts frames to event spikes in parallel.

### Step 1: Perception & Execution

- **CountGD**: Receives the initial intent and provides $N_{visible}$ (Visual Count).
- **SAM2 + Depth**: Receives the initial intent and provides the **3D Point Cloud**.
- **V-JEPA**: Maintains temporal memory state.
- **V2E**: Produces event energy (Spikes).

### Step 2: 3D Fusion & Conflict Detection

- **Fusion**: Menghapus (_mask subtraction_) area SAM2 dari Spike Map untuk mendeteksi **Residu Spike** (gerakan di luar area mask).
- **Math Analysis**: `MathUtils` memproses **Raw Point Cloud** dari SAM2 + DepthAnything:
  - Menerapkan rumus **Riemann Sums / Lattice Counting** untuk mendapatkan $N_{volumetric}$.
- **Logic Gate**: Memeriksa tiga jenis anomali (Hierarchical Counting):
  1. **Anomali Spasial**: Jika ada Residu Spike tinggi (Discovery).
  2. **Anomali Volumetrik**: Jika $N_{visible}$ tidak masuk dalam range $[N_{volume\_min}, N_{volume\_max}]$.
  3. **Physical Constraint Violation**: Jika estimasi jumlah melampaui batas fisik ruang (Sanity Check).

### Step 3: Targeted SLM $\to$ VL-JEPA (Sutradara)

1.  **Reasoning**: SLM menganalisis bukti dari `Logic Gate` (discrepancy math vs visual, atau residu spike).
2.  **Morphism**: SLM memberikan hipotesis dan instruksi spesifik ke **VL-JEPA (Director)**.
3.  **Intent Update**: VL-JEPA memperbarui `Persistent_Context`.
4.  **Looping (Bayesian Information Gain)**:
    - **State Interpolation**: Hasil dari SLM diproyeksikan ke frame _current_ menggunakan prediksi temporal V-JEPA (mengatasi latensi berpikir SLM).
    - **Parametric Rerun**:
    * **Refinement**: SAM2 dipicu menggunakan **PointBeam Methodology** (Depth Peak Prompting - lokal maxima pada depth map) untuk memisahkan tumpukan.
    - **Discovery**: VL-intent baru dijalankan pada koordinat residu spike.

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
