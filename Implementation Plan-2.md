# Implementation Plan-2: Glide-and-Count V2 (Detailed)

## 0. Infrastructure & Source Techs (Status: DOWNLOADED)

The following technologies have been verified in the `Techs/` directory and will serve as the source of truth for the V2 pivot:

- **Input Layer**: `Techs/v2e-master/v2e-master` ([SensorsINI/v2e](https://github.com/SensorsINI/v2e))
- **The Director**: `Techs/VL-JEPA-main/VL-JEPA-main` ([JosefAlbers/VL-JEPA](https://github.com/JosefAlbers/VL-JEPA))
- **The Brain**: `Techs/jepa-main/jepa-main` ([facebookresearch/jepa](https://github.com/facebookresearch/jepa))
- **The Executor**: `Techs/SAM2-main/SAM2-main` ([niki-amini-naieni/SAM2](https://github.com/niki-amini-naieni/SAM2))

---

## 1. Goal

Pivot the architecture from frame-based instance segmentation (V1: SAM2/DINOv2) to a hybrid event-driven world model (V2: v2e/VL-JEPA/V-JEPA 2).

## 2. Proposed Changes

### [Component] Input Layer: Event-Based Capture

Summary: Transition from RGB-dominated processing to synthetic event streams via `v2e`.

#### [NEW] [v2e_engine.py](file:///d:/Antigravity/Test%20VJEPA%20EVENTBASED%20LLM/Implementation/src/models/v2e_engine.py)

- Context: Wraps `Techs/v2e-master/v2e-master/v2ecore`.
- Function: Implements microsecond-resolution synthetic event generation from standard video to eliminate motion blur.

#### [MODIFY] [event_gen.py](file:///d:/Antigravity/Test%20VJEPA%20EVENTBASED%20LLM/Implementation/src/kernels/event_gen.py)

- Refactor existing kernels to support high-fidelity intensity changes as defined in `v2e`.

---

### [Component] The Director: Intent & Identification (VL-JEPA)

Summary: Use vision-language knowledge to autonomously identify scanning context.

#### [NEW] [vl_jepa_engine.py](file:///d:/Antigravity/Test%20VJEPA%20EVENTBASED%20LLM/Implementation/src/models/vl_jepa_engine.py)

- Context: Integrates `Techs/VL-JEPA-main/VL-JEPA-main/vljepa`.
- Function: Maps visual event patterns to semantic "sku_ids" without manual prompting.

---

### [Component] The Brain: World Model (V-JEPA 2)

Summary: Move from 2D temporal memory to a 3D persistent world model.

#### [NEW] [world_model_engine.py](file:///d:/Antigravity/Test%20VJEPA%20EVENTBASED%20LLM/Implementation/src/models/world_model_engine.py)

- Context: Uses `Techs/jepa-main/jepa-main/src` (V-JEPA Backbone).
- Function: Implements 3D spatial reasoning for object permanence through occlusions.

#### [DELETE] [temporal_memory.py](file:///d:/Antigravity/Test%20VJEPA%20EVENTBASED%20LLM/Implementation/src/models/temporal_memory.py)

- Superceded by the more advanced World Model logic.

---

### [Component] The Executor: Counting (SAM2)

Summary: Precise ID assignment and mathematical enumeration using Generalized Detection.

#### [NEW] [countgd_executor.py](file:///d:/Antigravity/Test%20VJEPA%20EVENTBASED%20LLM/Implementation/src/models/countgd_executor.py)

- Context: Integrates `Techs/SAM2-main/SAM2-main` (GroundingDINO Engine).
- Function: Performs temporal filtering and final tallying based on SAM2 logic.

---

### [Component] Pipeline & UI

Summary: Orchestrate the V2 flow and visualize spatial confidence.

#### [MODIFY] [engine.py](file:///d:/Antigravity/Test%20VJEPA%20EVENTBASED%20LLM/Implementation/src/pipeline/engine.py)

- Rewrite orchestrator to execute: `v2e -> VL-JEPA -> V-JEPA 2 -> SAM2`.

#### [MODIFY] [main.py](file:///d:/Antigravity/Test%20VJEPA%20EVENTBASED%20LLM/Implementation/main.py)

- Update visualization to show 3D spatial confidence maps and event boundaries.

---

## 3. Verification Plan

### Automated Tests

- **Kernel Fidelity**: Test `v2e_engine.py` against standard CMOS blur scenarios.
- **Permanence Check**: Script to verify ID retention in `world_model_engine.py` after 50% mask-out.

### Manual Verification

- **Glide Test**: Perform a rapid scan (approx 1.5m/s) and verify the final inventory JSON tally on an L40S instance.
