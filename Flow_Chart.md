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
    Start([Start Loop]) --> V2E[Receive Spike Events]
    V2E --> VJEPA_Enc[Encode V-JEPA Latents]
    VJEPA_Enc --> Perception[Perception Exec: CountGD + SAM2 + Depth]

    Perception --> LogicGate{Anomaly Detected?}

    LogicGate -- No (Spatial & Volume OK) --> End([Output Final Count])

    LogicGate -- Yes (Spatial Anomaly) --> Discovery[Discovery: Find New Target]
    LogicGate -- Yes (Volume Anomaly) --> Refinement[Refinement: Inspect Occlusion]

    Discovery --> SLM[SLM Hypothesizer]
    Refinement --> SLM

    SLM --> VLJEPA[Director: Update Intent]
    VLJEPA --> Interpolate[State Interpolation: V-JEPA Projection]
    Interpolate --> Start
```
