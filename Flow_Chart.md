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
    Start([Start Loop]) --> v2e_sensor_node[v2e_sensor_node: Spike Energy]
    Start --> vjepa_brain_node[vjepa_brain_node: Temporal Latent]

    vjepa_brain_node --> vljepa_director_node[vljepa_director_node: Intent]

    vljepa_director_node --> countgd_executor_node[countgd_executor_node: N_visible]
    vljepa_director_node --> sam2_depth_node[sam2_depth_node: 3D Point Cloud]

    countgd_executor_node --> fusion_engine_node[fusion_engine_node: Anomaly Calc]
    sam2_depth_node --> fusion_engine_node
    v2e_sensor_node --> fusion_engine_node

    fusion_engine_node --> logic_gate_node{Logic Gate Decision}

    logic_gate_node -- "Confident (Exit)" --> End([Final Inventory Audit])

    logic_gate_node -- "Anomaly (Loop)" --> targeted_slm_node[Targeted SLM: Reasoning]

    targeted_slm_node --> interpolation_node[interpolation_node: State Mapping]

    interpolation_node --> vljepa_director_node

    logic_gate_node -- "Max Loops" --> End
```
