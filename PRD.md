# Product Requirements Document (PRD): V-JEPA Event-Based Inventory Intelligence

## 1. Project Overview

**Code Name:** Inference Visualizer (Advanced)
**Goal:** Create a robust inventory counting system that combines rapid segmentation (SAM2), fine-grained instance recognition (DINOv2), density-based grouping (DBSCAN), and intelligent occlusion handling (V-JEPA Temporal Memory). The system eventually evolves into an "Audit Assistant" capable of natural language interaction (VLM).

## 2. Problem Statement

Inventory tracking in dynamic environments (warehouses, retail) suffers from:

1.  **Over-segmentation**: Items being split into too many categories.
2.  **Occlusion**: Items disappearing behind others or being blocked by hands.
3.  **Lack of Context**: Simple counters don't "understand" the scene (e.g., who took what?).

## 3. Core Technologies

- **Segmentation**: **SAM2 (Segment Anything Model 2)** with optimized sensitivity for high-quality product masks.
- **Feature Extraction**: **DINOv2 (ViT-Large)** for superior visual discrimination between identical-looking products.
- **Clustering**: **DBSCAN** for auto-detecting object types, augmented by a **Semantic Reasoning Layer** (Cosine Merge) to fix fragmentation.
- **Temporal Memory**: **V-JEPA (Frozen)** used for scene-based occlusion reasoning. It distinguishes between "Removed" and "Hidden" by comparing scene features.
- **VLM Head (Future)**: **Qwen2.5-VL** or similar for "Chat with Video", auto-labeling, and anomaly detection.
- **Infrastructure**: RunPod L40S for real-time inference.

## 4. Functional Requirements

### 4.1. Perception & Counting Pipeline

1.  **Strict Segmentation**: Filter masks by IOU (0.8) and Stability (0.85) to minimize noise.
2.  **Fine-Grained Re-ID**: Use 1024-dim DINOv2 embeddings to differentiate items.
3.  **Intelligent Clustering**:
    - Auto-detect types via DBSCAN.
    - **Reasoning Step**: Merge clusters with >0.85 visual similarity to prevent over-counting.
4.  **Occlusion Reasoning**: Use V-JEPA features to maintain "Memory" of items that are visually lost but likely still present in the box.

### 4.2. VLM Interaction (Audit Head)

1.  **Auto-Labeling**: Convert "Type 0" to "Brand X Milk" using VLM vision-language capability.
2.  **Question Answering**: Allow users to query the video (e.g., "Was anything dropped?").
3.  **Narrative Reporting**: Generate human-readable audit summaries.

### 4.3. Visualization

- Side-by-side view: Original RGB with Masks + Live Inventory Table (Visible vs. Memory).
- Saliency/Attention overlays from the V-JEPA backbone.

## 5. Success Metrics

- [x] Successfully merge fragmented clusters using Semantic Reasoning.
- [x] Correctly count 4/4 items in scanning video despite occlusion.
- [x] Activate Recursive Intent (Hybrid Path) for real-time anomaly handling.
- [ ] Achieve <5% counting error on high-density product videos.
- [ ] VLM successfully identifies specific brands without manual labeling.

## 6. Roadmap

1.  **Phase 1: Robust Perceiver (DONE)**: SAM2 + DINOv2 + DBSCAN + Reasoning Layer.
2.  **Phase 2: Occlusion Mastery (DONE)**: V-JEPA Temporal Memory + Depth V2.
3.  **Phase 3: Recursive Intent Activation (DONE)**: Hybrid Logic Gate + Targeted SLM.
4.  **Phase 4: VLM Expansion (NEXT)**: Integrate Qwen2.5-VL for "Chat with Video".
