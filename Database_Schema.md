"""
Database Schema

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol : DatabaseSchema (this document)

Non-Terminals :
┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
│ <InventoryTable> → Object state storage │
│ <MemoryTable> → Latent storage │
└───────────────────────────────────────────────────────────────────────────┘

Terminals : UUID, Integer, String, Float, JSON, Blob

Production Rules:
DatabaseSchema → <InventoryTable> <MemoryTable>
═══════════════════════════════════════════════════════════════════════════════
"""

# Database Schema: Inventory & Memory

The Antigravity V2 system uses a lightweight local storage for inventory states and temporal memory. In production, this can be mapped to a PostgreSQL/MongoDB instance.

## 1. Inventory & Track State (`inventory_items`)

| Column              | Type   | Description                                             |
| ------------------- | ------ | ------------------------------------------------------- |
| `id`                | UUID   | Unique identifier for the object instance.              |
| `label`             | String | Semantic label (e.g., "Brand X Milk").                  |
| `rho`               | Float  | Predicted density from DINOv2 specularity.              |
| `golden_alpha`      | Float  | Calibrated Alpha parameter for this object type.        |
| `rho_confidence`    | Float  | MLP prediction confidence.                              |
| `shield_scores`     | JSON   | {spatial, volumetric, latent} scores from Multi-Shield. |
| `fusion_confidence` | Float  | Final weighted consensus score.                         |
| `bbox`              | JSON   | [x, y, w, h] of the last detection.                     |
| `last_seen`         | Float  | Last relative timestamp seen.                           |

## 2. Temporal Context (`latent_memory`)

| Column        | Type    | Description                                        |
| ------------- | ------- | -------------------------------------------------- |
| `session_id`  | String  | Re-loadable session identifier.                    |
| `buffer_size` | Integer | Current size of the PersistentLatentContext deque. |
| `latent_vec`  | Blob    | 1024-dim context embedding from V-JEPA.            |
| `timestamp`   | Float   | Relative timestamp in milliseconds.                |
