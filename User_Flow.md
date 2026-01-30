# User Flow: Inventory Audit Assistant

The end-user (Auditor) interacts with the system through a high-level natural language interface.

```mermaid
graph LR
    User([Auditor]) --> Prompt["'Audit the cups'"]
    Prompt --> vljepa_director_node[Identify Intent]
    vljepa_director_node --> Perception[v2e + SAM2 + Depth + CountGD]
    Perception --> logic_gate_node{Check Anomaly}
    logic_gate_node -- "Confident" --> Resolved[Final Audit Report]
    logic_gate_node -- "Anomaly" --> targeted_slm_node[Self-Reflection]
    targeted_slm_node -- "New Intent" --> vljepa_director_node
    Resolved --> User
```

## 1. Modes of Interaction

- **Passive Monitoring**: The system counts silently and flags anomalies.
- **Active Querying**: User asks "Where did the Red Bull box go?" and the system highlights the last known/predicted location.
