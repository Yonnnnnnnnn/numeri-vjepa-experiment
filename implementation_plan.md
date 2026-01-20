# Implementation Plan: Robust V-JEPA & VLM Pipeline

## 1. Goal

Implement a robust inventory intelligence pipeline utilizing SAM2, DINOv2, and V-JEPA, followed by a VLM-based "Audit Assistant" (Status: Phase 1 & 2 Complete).

## 2. Core Architecture (Implementation Phase)

### 2.1. Perception Layer (Status: DONE)

- **Segmentation**: `Implementation/src/models/segmentation_engine.py` using SAM2.
- **Embeddings**: `Implementation/src/models/embedding_engine.py` using DINOv2 (ViT-Large).
- **Clustering**: `Implementation/src/models/clustering_engine.py` using DBSCAN.

### 2.2. Robustness Logic (Status: DONE)

- **SAM2 Optimization**: Tightened thresholds (`IOU=0.8`, `Stability=0.85`) to filter noise.
- **Semantic Reasoning**: Post-DBSCAN merge step based on **Cosine Similarity > 0.85**.
- **Temporal Memory**: `Implementation/src/models/temporal_memory.py` using V-JEPA scene features to handle occlusion.

### 2.3. VLM Layer (Status: TODO - NEXT)

**Goal:** Integrate Qwen2.5-VL for natural language interaction.

- **Component**: `Implementation/src/models/vlm_engine.py` (NEW).
- **Functionality**:
  - **Auto-Labeling**: Map `Type X` to text labels using the most representative crop from the Clustering Engine.
  - **Session Chat**: Maintain a video summary and answer user questions.

## 3. Actual File Structure

- `Implementation/`
  - `src/`
    - `models/`
      - `segmentation_engine.py`: SAM2 wrapper.
      - `embedding_engine.py`: DINOv2 wrapper.
      - `clustering_engine.py`: DBSCAN + Reasoning Layer.
      - `temporal_memory.py`: V-JEPA Occlusion logic.
      - `vlm_engine.py`: (Planned) Qwen2.5-VL interface.
    - `pipeline/`
      - `engine.py`: Main inference loop orchestrator.
  - `main.py`: Entry point with visualization logic.
  - `scripts/`: Download and test utilities.

## 4. Verification & Roadmap

### Phase 1 & 2 (Verified)

- [x] **Clustering Verification**: Confirmed Semantic Merge reduces 11 types to 4 types.
- [x] **Segmentation Sensitivity**: Verified high-IOU filtering removes background noise.
- [x] **Occlusion Test**: Verified V-JEPA maintains count when items are hidden.

### Phase 3: VLM Expansion (Planned)

1.  **Model Setup**: Download Qwen2.5-VL weights to `Techs/`.
2.  **Prompt Development**: Structure prompts for item identification and anomaly detection.
3.  **Chat Interface**: Integrate with CLI/UI for interactive audit queries.
