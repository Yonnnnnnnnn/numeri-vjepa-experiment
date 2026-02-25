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
    participant G as Genesis (Step 0: VLMxScout)
    participant S as Sensors (V2E/SAM2/Density)
    participant B as Brain (V-JEPA)
    participant D as Director (VL-JEPA)
    participant M as Math (V3 Reconciliation)
    participant F as Fusion (Multi-Shield)
    participant LG as Gate (Logic Gate)
    participant R as Reasoner (SLM)

    Note over G: Phase 0: Semantic Setup
    G->>D: genesis_intents & reference_crops

    Note over B,D: Phase 1: Anchor
    B->>D: Latent Context & Memory
    D->>S: Step 2-3: Target Intent & Dynamic Bounds

    Note over S: Phase 2: Orbit (Discovery)
    S->>M: Step 4-7: Per-Cluster Volumes & Point Clouds
    S->>F: Step 4-7: Visible Counts & Spikes

    Note over M,F: Phase 2: Audit
    M->>F: Step 8-9: N_volumetric consensus
    F->>LG: Step 10: Multi-Shield Confidence

    Note over LG,R: Phase 4: Reconciliation
    LG->>EXIT: Step 11: Valid Consensus
    LG->>R: Step 12: Anomaly / Confusion
    R->>D: Step 13: Intent Loop / Calibration Update
```
