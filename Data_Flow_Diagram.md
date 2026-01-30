"""
Data Flow Diagram

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol : DataFlowDiagram (this document)

Non-Terminals :
┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
│ <Sequence> → Data flow between modules │
└───────────────────────────────────────────────────────────────────────────┘

Terminals : Mermaid, sequenceDiagram, participant

Production Rules:
DataFlowDiagram → <Sequence>
═══════════════════════════════════════════════════════════════════════════════
"""

# Data Flow Diagram: Spike-to-Count Transformation

This diagram details the transformation of raw event data into actionable inventory intelligence.

```mermaid
sequenceDiagram
    participant S as Sensor (v2e)
    participant B as Brain (V-JEPA)
    participant D as Director (VL-JEPA)
    participant E as Executor (CountGD + SAM2 + Depth)
    participant L as Logic Gate (Math Guard)
    participant R as Razonamiento (Targeted SLM)

    S->>L: Spike Map (Residual Energy)
    B->>E: Latent Context
    D->>E: Semantic Intent
    E->>L: N_visible & N_volumetric
    L->>R: Trigger reasoning on anomaly
    R->>D: Hipotesis / Refined Intent
    L->>User: Final Count
```
