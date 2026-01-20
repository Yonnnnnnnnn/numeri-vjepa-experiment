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
    VJEPA_Enc --> SAM2[Segment Current Frame]
    SAM2 --> DINO[Extract DINOv2 Features]
    DINO --> DBSCAN[Cluster Objects]

    DBSCAN --> OccCheck{Check for Missing Items?}
    OccCheck -- Yes --> VJEPA_Pred[Predict Future States]
    VJEPA_Pred --> RecMatch[Reconcile Memory with Masks]
    RecMatch --> UpdateInv[Update Inventory State]

    OccCheck -- No --> UpdateInv
    UpdateInv --> VLM_Audit[VLM Audit Head]
    VLM_Audit --> Anomaly{Anomaly Detected?}

    Anomaly -- Yes --> RecIntent[Trigger Recursive Intent]
    RecIntent --> VLJEPA[Update Director Instructions]
    VLJEPA --> Start

    Anomaly -- No --> End([Output Final Count])
```
