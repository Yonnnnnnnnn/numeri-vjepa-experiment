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
    participant E as Executor (CountGD)
    participant R as Reflection (FusionEngine)

    S->>B: Spike Stream (Temporal-Visual Data)
    B->>D: Latent Context (Spatial Memory)
    D->>E: Semantic Intent ("Count Object X")
    B->>E: Visual Patches (Cropped Features)
    E->>R: Raw Counts & Bboxes
    R->>D: Anomaly Detection (Recursive Intent Trigger)
    D->>E: Refined Intent (Self-Correction)
    E->>User: Final Inventory Audit
```
