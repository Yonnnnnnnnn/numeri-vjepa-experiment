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
    LangGraph("LangGraph<br>(State Container)")
    V2E("V2E<br>(Spike Source)")
    VJEPA("V-JEPA<br>(Temporal Context)")
    VLJEPA("VL-JEPA<br>(Intent Director)")
    S2DA("SAM2 + DepthAnything<br>(3D Perception Kernel)")
    Fusion("Fusion Engine<br>(Anomaly Calc)")
    LogicGate("Logic Gate<br>(Decision Guard)")
    SLM("Targeted SLM<br>(Mathematical Reasoner)")
    MathUtils("MathUtils<br>(Kernel)")

    %% Define Morphisms (Arrows = Actions)

    %% Sensory & Intent Flow
    VJEPA -- "1. Provides Temporal Latent" --> Fusion
    V2E -- "2. Feeds Residual Spikes" --> Fusion
    VLJEPA -- "3. Sets Domain Intent" --> S2DA

    %% Perception & Math Flow
    S2DA -- "4. Generates 3D Point Cloud" --> MathUtils
    MathUtils -- "5. Calculates Volume/Count" --> LogicGate
    S2DA -- "6. Report Visual Masks" --> Fusion

    %% Analysis & Decision Flow
    Fusion -- "7. Outputs Anomaly Score" --> LogicGate
    LogicGate -- "8. Submits Physical Evidence" --> SLM

    %% Recursion & Exit
    SLM -- "9. Confirms Count/Refinement" --> LogicGate
    LogicGate -- "10. Updates Global State" --> LangGraph
    SLM -- "11. Updates Context" --> VLJEPA

    %% Orchestration
    LangGraph -. "Orchestrate" .-> VJEPA
    LangGraph -. "Orchestrate" .-> VLJEPA
    LangGraph -. "Orchestrate" .-> S2DA

    %% Styling
    linkStyle default stroke-width:2px,fill:none,stroke:white;
    classDef object fill:#1a1a1a,stroke:#4a4a4a,stroke-width:2px,color:white;
    class LangGraph,V2E,VJEPA,VLJEPA,S2DA,Fusion,LogicGate,SLM,MathUtils object;
```
