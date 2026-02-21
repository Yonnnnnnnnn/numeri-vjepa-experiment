# Product Requirements Document (PRD): V3.1 Sovereignty Protocol

## 1. Project Overview

**Code Name**: Sovereignty Engine
**Goal**: Transition from a heuristic counting pipeline to a recursive, physics-aligned "Sovereignty" system. The system treats volumetric data ($N_{vol}$) as the ground truth "Sovereign" evidence, using LangGraph recursive loops to reconcile discrepancies with visual detection ($N_{vis}$).

## 2. Problem Statement (V3.1)

1.  **Counting Noise**: Simple 2D visual counts are unreliable in deep stacks.
2.  **Calibration Drift**: Unit volume ($V_{\mu}$) and concavity ($\alpha$) vary per object type.
3.  **Ambiguity**: Visual occlusions require temporal memory and physical proof to resolve.

## 3. Core Technologies (V3.1 Upgrade)

- **Geometric Kernel**: `AlphaShape` 3D Hull generation with **Golden Alpha Calibration**.
- **Density Engine**: DINOv2 CLS-features + MLP Regressor for packing density ($\rho$) prediction.
- **Sovereignty Chain**: **LangGraph** implementation of the 13-step recursive protocol.
- **Multi-Shield Fusion**: Triangulation of Spatial (V2E), Volumetric (AlphaHull), and Latent (V-JEPA) shields.

## 4. Functional Requirements

### 4.1. Sovereignty & Reconciliation

1.  **Volumetric Dominance**: If $N_{vol} > N_{vis} + \text{Threshold}$, trigger a **Discovery Loop** (SLM Hypothesis).
2.  **Golden Alpha Calibration**: Auto-calibrate Hull concavity when a single object is isolated.
3.  **Density Sensing**: Adapt counting math based on visual specularity/texture of the item.

### 4.2. Intent & Discovery

1.  **Dynamic Intent**: The Director (`vljepa_director_node`) can add new classes to search based on SLM feedback.
2.  **Physical Calibration**: Auto-refine $V_{\mu}$ when visual count is highly confident.

### 4.3. Defensive Processing (V3.8)

1.  **Tensor Standardization**: Mandatory 224x224 resizing for all V-JEPA context inputs to ensure batching/temporal compatibility.
2.  **Agnostic Box Parsing**: Support for variant GroundingDINO output formats (list/dict) with normalized coordinate mapping.
3.  **Label Sovereignty**: Strict punctuation-aware filtering in VLM analyst to prevent instruction text pollution in the SKU manifest.

## 5. Success Metrics

- [x] Integrate 3D AlphaHull for volumetric estimation.
- [x] Implement 13-step recursive logic gate in LangGraph.
- [x] Reach consensus on 5+ item stacks with <10% error.
- [x] **V3.8 Stability**: Achieve 0 count of `RuntimeError: stack expects each tensor to be equal size` during 10-minute video processing.

## 6. Roadmap

1.  **Phase 1-4 (DONE)**: Core Perception, Memory, Density, and Fusion.
2.  **Phase 5 (DONE)**: 13-Step Orchestration (Sovereignty Protocol).
3.  **Phase 6 (IN PROGRESS)**: Defensive Refinement, Cleanup, & Engine Optimization. V3.8 completed (Tensor/Parse Fixes).
