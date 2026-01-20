# User Flow: Inventory Audit Assistant

The end-user (Auditor) interacts with the system through a high-level natural language interface.

```mermaid
graph LR
    User([Auditor]) --> Prompt["'Audit the pallet of milk'"]
    Prompt --> System[V2 System Starts]
    System --> Live["Visual Feedback (SAM2 Overlays)"]
    Live --> Audit["Live Inventory Count"]
    Audit --> Anomaly["System Detects Occlusion/Discrepancy"]
    Anomaly --> Notify["'I lost track of a item, checking memory...'"]
    Notify --> Resolved["Final Report Generated"]
    Resolved --> User
```

## 1. Modes of Interaction

- **Passive Monitoring**: The system counts silently and flags anomalies.
- **Active Querying**: User asks "Where did the Red Bull box go?" and the system highlights the last known/predicted location.
