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

# Architecture Diagram: Antigravity V2

The system follows a "Tangled Hierarchy" (Strange Loop) architecture where high-level semantic intent and low-level sensory data influence each other recursively.

```mermaid
graph TB
    subgraph Sensory_Layer [Sensory Layer (Mata)]
        V2E[v2e: Event-Based Camera]
        Spikes[Spike Train / Event Frames]
    end

    subgraph Brain_Layer [Brain Layer (Otak)]
        VJEPA[V-JEPA: Temporal Memory]
        Latents[Latent Representations]
    end

    subgraph Logic_Layer [Logic Layer (Intent)]
        VLJEPA[VL-JEPA: Director]
        Intent[Semantic Intent / Instructions]
    end

    subgraph Action_Layer [Action Layer (Executor)]
        CountGD[CountGD: Bounding Box & Counting]
        Results[Inventory Results]
    end

    V2E --> Spikes
    Spikes --> VJEPA
    VJEPA --> Latents
    Latents --> VLJEPA
    VLJEPA --> Intent
    Intent --> CountGD
    Latents --> CountGD
    CountGD --> Results

    %% Strange Loops
    Results -.->|Recursive Intent| VLJEPA
    Latents -.->|Sensory-Predictive| V2E
    Intent -.->|Semantic-Kinetic| VJEPA
```
