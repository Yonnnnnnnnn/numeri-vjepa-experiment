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
    %% Phase 0: Anchor
    Start([Start Loop]) -- "Step 1" --> vjepa_brain_node[vjepa_brain_node: Temporal Context]
    vjepa_brain_node -- "Step 2-3" --> vljepa_director_node[vljepa_director_node: Intent & Calibration]

    %% Phase 1: Orbit
    vljepa_director_node -- "Step 4" --> countgd_executor_node[countgd_executor_node: N_visible]
    vljepa_director_node -- "Step 5" --> v2e_sensor_node[v2e_sensor_node: Spike Residue]
    vljepa_director_node -- "Step 6" --> sam2_depth_node[sam2_depth_node: 3D Geometry]
    vljepa_director_node -- "Step 7" --> density_sensing_node[density_sensing_node: Rho Prediction]

    %% Phase 2: Audit
    sam2_depth_node & density_sensing_node -- "Step 8" --> v3_math_node[v3_math_node: N_volumetric]
    v3_math_node & countgd_executor_node & v2e_sensor_node -- "Step 9" --> fusion_engine_node[fusion_engine_node: Multi-Shield Fusion]

    %% Phase 3: Reconciliation
    fusion_engine_node -- "Step 10" --> logic_gate_node{Logic Gate Evaluator}

    logic_gate_node -- "Step 11: Valid (Exit)" --> End([Final Inventory Audit])

    logic_gate_node -- "Step 12: Anomaly / Discovery" --> targeted_slm_node[Targeted SLM: Reasoner]
    targeted_slm_node -- "Step 13: Loop" --> vljepa_director_node

    %% Failsafe
    logic_gate_node -- "Max Loops" --> End
```
