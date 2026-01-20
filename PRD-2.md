# Product Requirements Document (PRD-2): Glide-and-Count V2

## 1. Project Overview

**Code Name:** Glide-and-Count (2026 Architecture)
**Goal:** Implement a hybrid event-driven world model for warehouse inventory counting that eliminates motion blur, handles extreme lighting, and maintains 100% object permanence during high-speed "gliding" scans.

## 2. Problem Statement

Current frame-based systems fail in real-world warehouse scenarios because:

1.  **Motion Blur**: Moving a camera quickly while scanning labels causes blurring that breaks traditional OCR and detection.
2.  **Lighting & Shadows**: Harsh warehouse lighting or dark corners create high-contrast scenes that lose detail in standard RGB.
3.  **Spatial Amnesia**: Disappearing and reappearing objects often get double-counted or lost in memory.

## 3. Core Technologies (V2 Architecture)

- **Input Layer (Synthetic Event Stream)**:
  - **v2e Engine**: Converts standard CMOS video into microsecond-resolution event streams (ON/OFF pixel changes).
  - **Advantage**: Zero motion blur, high dynamic range (HDR), and reduced data redundancy.
- **The Director (VL-JEPA)**:
  - **Role**: Vision-Language Intent.
  - **Function**: Autonomously identifies the "concept" (e.g., "Heineken Bottle") from the event stream without manual prompts.
- **The Brain (V-JEPA 2 World Model)**:
  - **Role**: Spatial Reasoning & Permanence.
  - **Function**: Builds a 3D "World Model" of the shelf. It "remembers" objects behind occlusions or outside the current field of view.
- **The Executor (CountGD)**:
  - **Role**: Tracking & Enumeration (Generalized Detection).
  - **Function**: Performs the mathematical counting using temporal filtering.
  - **Source**: [niki-amini-naieni/CountGD](https://github.com/niki-amini-naieni/CountGD)

## 4. Functional Requirements

### 4.1. Event-Driven Perception

1.  **High-Speed Processing**: The pipeline must handle input from rapid camera movement without segmentation failure via **v2e** ([SensorsINI/v2e](https://github.com/SensorsINI/v2e)).
2.  **Autonomous Labeling (VL-JEPA)**: The system must "understand" the context via **VL-JEPA** ([JosefAlbers/VL-JEPA](https://github.com/JosefAlbers/VL-JEPA)).

### 4.2. Spatial Intelligence

1.  **3D Permanence**: Assign persistent IDs to objects in 3D space using **V-JEPA 2 World Model**.
2.  **Occlusion Mastery**: If an item is 80% blocked, V-JEPA 2 must still maintain its existence in the count.

### 4.3. Accurate Enumeration

1.  **Temporal Filtering**: Reject "ghost" detections that appear for less than 100ms.
2.  **Global Tally**: Provide a single, final inventory count confirmed across the entire "glide" path.

## 5. Success Metrics (Target 2026)

- **Speed**: Scan at 2.5m/s (Walking speed) with 99.5% accuracy.
- **Efficiency**: 2.85x more efficient than 2024 frame-based architectures.
- **Occlusion**: 100% ID retention through partial/full temporary occlusion.
- **Hardware**: Runs in real-time on high-performance mobile edge devices (NVIDIA L40S equivalent in size).

## 6. Architecture Comparison (V1 vs V2)

| Feature      | V1 (SAM2 + DINOv2)       | V2 (Event-Driven World Model)     |
| :----------- | :----------------------- | :-------------------------------- |
| **Input**    | RGB (Vulnerable to blur) | **v2e Engine (Synthetic Events)** |
| **Director** | Manual Prompting         | **VL-JEPA (Autonomous Intent)**   |
| **Brain**    | 2D Temporal Memory       | **3D World Model (V-JEPA 2)**     |
| **Executor** | DBSCAN / Cosine Merge    | **CountVid (Temporal Filtering)** |

## 7. Preliminary Implementation Map

- **Event Capture**: `v2e_engine.py` & `event_gen.py`.
- **Identification**: `vl_jepa_engine.py`.
- **Spatial Reasoning**: `world_model_engine.py`.
- **Calculation**: `countvid_executor.py`.
- **Orchestration**: `pipeline/engine.py` (V2 Mode).
