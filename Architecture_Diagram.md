"""
Architecture Diagram

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol : ArchitectureDiagram (this document)

Non-Terminals :
┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
│ <SensoryLayer> → v2e integration │
│ <BrainLayer> → V-JEPA integration │
│ <LogicLayer> → VL-JEPA integration │
│ <ActionLayer> → CountVid integration │
└───────────────────────────────────────────────────────────────────────────┘

Terminals : Mermaid, subgraph, graph

Production Rules:
ArchitectureDiagram → <SensoryLayer> <BrainLayer> <LogicLayer> <ActionLayer>
═══════════════════════════════════════════════════════════════════════════════
"""

# Architecture Diagram: Recursive Intent Category Theory

The following diagram represents the system as a Category where **Objects** are system states/components and **Morphisms** are the functional transformations between them.

```mermaid
graph TD
    %% Define Objects (Nodes)
    vjepa_brain_node("vjepa_brain_node<br>(Temporal Context)")
    vljepa_director_node("vljepa_director_node<br>(Intent & Calibration)")
    v2e_sensor_node("v2e_sensor_node<br>(Residue Guard)")
    sam2_depth_node("sam2_depth_node<br>(3D Geometry)")
    density_sensing_node("density_sensing_node<br>(Density Sensing)")
    countvid_executor_node("countvid_executor_node<br>(Visual Count)")
    v3_math_node("v3_math_node<br>(Volumetric Reconciliation)")
    fusion_engine_node("fusion_engine_node<br>(Multi-Shield Fusion)")
    logic_gate_node("logic_gate_node<br>(Sovereignty Guard)")
    targeted_slm_node("targeted_slm_node<br>(Reasoning Hook)")

    %% Define Sovereignty Flow (13-Step)
    START["START"] -- "Step 1" --> vjepa_brain_node
    vjepa_brain_node -- "Step 2-3" --> vljepa_director_node
    vljepa_director_node -- "Step 4-7: Orbit" --> v2e_sensor_node
    vljepa_director_node -- "Step 4-7: Orbit" --> sam2_depth_node
    vljepa_director_node -- "Step 4-7: Orbit" --> density_sensing_node
    vljepa_director_node -- "Step 4-7: Orbit" --> countvid_executor_node

    v2e_sensor_node -- "Evidence" --> fusion_engine_node
    sam2_depth_node -- "Evidence" --> v3_math_node
    density_sensing_node -- "Evidence" --> v3_math_node
    countvid_executor_node -- "Evidence" --> fusion_engine_node

    v3_math_node -- "Step 8-9: N_vol" --> fusion_engine_node
    fusion_engine_node -- "Step 10: Confidence" --> logic_gate_node
    logic_gate_node -- "Step 11: Evaluate" --> EXIT((EXIT))
    logic_gate_node -- "Step 12: Anomaly" --> targeted_slm_node
    targeted_slm_node -- "Step 13: Loop" --> vljepa_director_node

    %% Styling
    linkStyle default stroke-width:2px,fill:none,stroke:white;
    classDef object fill:#1a1a1a,stroke:#4a4a4a,stroke-width:2px,color:white;
    class vjepa_brain_node,vljepa_director_node,v2e_sensor_node,sam2_depth_node,density_sensing_node,countvid_executor_node,v3_math_node,fusion_engine_node,logic_gate_node,targeted_slm_node object;
```
