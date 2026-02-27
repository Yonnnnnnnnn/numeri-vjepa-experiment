"""
Flow Chart

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol : FlowChart (this document)

Non-Terminals :
┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
│ <InferenceLoop> → Real-time vision pipeline │
└───────────────────────────────────────────────────────────────────────────┘

Terminals : Mermaid, flowchart, graph

Production Rules:
FlowChart → <InferenceLoop>
═══════════════════════════════════════════════════════════════════════════════
"""

# Flow Chart: Inference & Occlusion Reasoning

Logic flow for the real-time vision pipeline.

```mermaid
flowchart TD
    %% Phase 0: Semantic Setup
    Start([Start Loop]) --> check_init{Is Initialized?}
    check_init -- "No" --> saccade[Step 0.1: Peripheral Saccade (DINOv2 Saliency)]
    saccade --> fixation[Step 0.2: PointBeam Fixation (Hotspot Discovery)]
    fixation --> fovea[Step 0.3: Foveated Genesis (Unified VLM Qwen)]
    fovea --> lnn_intent_filter[LNN: Intent Filter]
    lnn_intent_filter -- "Validated SKU" --> vjepa_brain_node

    check_init -- "Yes" --> vjepa_brain_node

    vjepa_brain_node[vjepa_brain_node: Temporal Context] -- "Step 1" --> latent_director_node[latent_director_node: Latent & Calibration]

    %% Phase 1: Orbit
    latent_director_node -- "Step 4" --> countgd_executor_node[countgd_executor_node: N_visible]
    latent_director_node -- "Step 5" --> v2e_sensor_node[v2e_sensor_node: Spike Residue]
    latent_director_node -- "Step 6" --> sam2_depth_node[sam2_depth_node: 3D Geometry]
    latent_director_node -- "Step 7" --> density_sensing_node[density_sensing_node: Rho Prediction]

    %% Phase 2: Audit
    sam2_depth_node & density_sensing_node -- "Step 8" --> v3_math_node[v3_math_node: Volumetric Pooling]
    v3_math_node --> lnn_identity_arbitrator[LNN: Identity Arbitrator (Latent vs Volume)]
    lnn_identity_arbitrator -- "N_volumetric" --> fusion_engine_node[fusion_engine_node: Multi-Shield Fusion]
    countgd_executor_node & v2e_sensor_node --> fusion_engine_node

    %% Phase 3: Reconciliation
    fusion_engine_node -- "Step 10" --> logic_gate_node{Logic Gate Evaluator}

    logic_gate_node -- "Step 11: Valid (Exit)" --> End([Final Inventory Audit])

    logic_gate_node -- "Step 12: Anomaly / Discovery" --> targeted_slm_node[Targeted SLM: Reasoner]
    targeted_slm_node -- "Step 13: Loop" --> latent_director_node

    %% Failsafe
    logic_gate_node -- "Max Loops" --> End
```
