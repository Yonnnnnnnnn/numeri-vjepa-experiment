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

## 1. Inventory State (`inventory_items`)

| Column        | Type      | Description                                           |
| ------------- | --------- | ----------------------------------------------------- |
| `id`          | UUID      | Unique identifier for the object instance.            |
| `type_id`     | Integer   | ID assigned by DBSCAN/VL-JEPA.                        |
| `label`       | String    | Semantic label (e.g., "Brand X Milk").                |
| `confidence`  | Float     | Confidence score from the Executor.                   |
| `bbox`        | JSON      | [x, y, w, h] of the last detection.                   |
| `last_seen`   | Timestamp | Last time the object was visually matched.            |
| `is_occluded` | Boolean   | True if inferred by V-JEPA but not currently visible. |

## 2. Temporal Memory (`latent_memory`)

| Column       | Type    | Description                            |
| ------------ | ------- | -------------------------------------- |
| `frame_id`   | Integer | Sequential frame ID.                   |
| `latent_vec` | Blob    | 1024-dim latent embedding from V-JEPA. |
| `timestamp`  | Float   | Event timestamp in milliseconds.       |
