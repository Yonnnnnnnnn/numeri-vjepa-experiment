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
│ <ActionLayer> → CountGD integration │
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
    v2e_sensor_node("v2e_sensor_node<br>(Spike Source)")
    vjepa_brain_node("vjepa_brain_node<br>(Temporal Context)")
    vljepa_director_node("vljepa_director_node<br>(Intent Director)")
    countgd_executor_node("countgd_executor_node<br>(Visual Executor)")
    sam2_depth_node("sam2_depth_node<br>(3D Perception Kernel)")
    fusion_engine_node("fusion_engine_node<br>(Anomaly Calc)")
    logic_gate_node("logic_gate_node<br>(Decision Guard)")
    targeted_slm_node("targeted_slm_node<br>(Mathematical Reasoner)")
    MathUtils("MathUtils<br>(Kernel)")

    %% Define Morphisms (Arrows = Actions)

    %% Sensory & Intent Flow
    vjepa_brain_node -- "1. Provides Temporal Latent" --> fusion_engine_node
    v2e_sensor_node -- "2. Feeds Residual Spikes" --> fusion_engine_node
    vljepa_director_node -- "3. Sets Domain Intent" --> sam2_depth_node
    vljepa_director_node -- "3. Sets Domain Intent" --> countgd_executor_node

    %% Perception & Math Flow
    sam2_depth_node -- "4. Generates 3D Point Cloud" --> MathUtils
    MathUtils -- "5. Calculates Volume/Count" --> logic_gate_node
    sam2_depth_node -- "6. Report Visual Masks" --> fusion_engine_node
    countgd_executor_node -- "7. Visible Count" --> fusion_engine_node

    %% Analysis & Decision Flow
    fusion_engine_node -- "8. Outputs Anomaly Score" --> logic_gate_node
    logic_gate_node -- "9. Submits Physical Evidence" --> targeted_slm_node

    %% Recursion & Exit
    targeted_slm_node -- "10. Updates Context" --> vljepa_director_node
    logic_gate_node -- "11. Accept/Exit" --> END["END"]

    %% Styling
    linkStyle default stroke-width:2px,fill:none,stroke:white;
    classDef object fill:#1a1a1a,stroke:#4a4a4a,stroke-width:2px,color:white;
    class v2e_sensor_node,vjepa_brain_node,vljepa_director_node,countgd_executor_node,sam2_depth_node,fusion_engine_node,logic_gate_node,targeted_slm_node,MathUtils object;
```
