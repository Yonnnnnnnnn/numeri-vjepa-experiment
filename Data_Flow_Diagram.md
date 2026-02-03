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
    participant S2 as SLM (Reasoning)

    S->>B: Spike Stream (Temporal-Visual Data)
    B->>D: Latent Context (Spatial Memory)
    D->>E: Semantic Intent ("Count Object X")
    B->>E: Visual Patches (Cropped Features)
    E->>R: Raw Counts & Bboxes
    R->>S2: Anomaly Data (Surprise Trigger)
    S2->>D: Refined Intent & Auto-Calibration
    D->>E: Refined Intent (Internal Update)
    E->>User: Final Inventory Audit
```
