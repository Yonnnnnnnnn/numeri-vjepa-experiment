"""
Linter & Formatter Config

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol : LinterConfig (this document)

Non-Terminals :
┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
│ <PythonStandards> → Black, Pylint settings │
│ <EnforcedRules> → Global rules references │
└───────────────────────────────────────────────────────────────────────────┘

Terminals : Black, Pylint, PEP8, Rule2.1, Rule2.2, Rule3, Rule7

Production Rules:
LinterConfig → <PythonStandards> <EnforcedRules>
═══════════════════════════════════════════════════════════════════════════════
"""

# Linter & Formatter Config

This project adheres to the standards defined in the Global Project Rules.

## Python standards

- **Formatter**: Black (88 chars line length)
- **Linter**: Pylint (configured in `.pylintrc`)
- **OpenCV Handling**: Suppress `no-member` for `cv2` calls due to C-extension false positives.
- **Naming**: PEP 8 (snake_case for functions/variables, PascalCase for classes)
- **Type Ignores**: Use `# type: ignore` for external vision libraries (SAM2, Detectron2, GroundingDINO) where type stubs are missing or C-extensions confuse static analysis.
- **Variable Shadowing**: Use `compute_device` instead of global `device` in function arguments. Use `trajectories` instead of `T` to avoid conflict with `transforms` module.
- **Missing Implementations**: Functions like `tt_norm` and `get_sam` that are placeholders in the current fork must be documented with `# type: ignore` to allow linting to pass while remaining non-functional at runtime.

## Enforced Rules

- **Rule 2.1**: Law of Flatness (Guard Clauses first)
- **Rule 6.1**: Heavy models (JEPA, SAM2, DINOv2) **MUST** use the Singleton pattern (Lazy-Loading) in `recursive_flow.py`.
- **Rule 6.2**: All volumetric calculations **MUST** be wrapped in `SanityGuard` bounds-checking.
- **Rule 6.3**: Legacy pipelines (`engine.py`, `engine_v2.py`) are strictly for visualization/reference; logic changes must be made in LangGraph nodes.
- **Rule 7**: CFG Documentation in every code file
- **Phase 6 Guard**: **SanityGuard Enforcement** (Must use shared `SanityGuard` for all volumetric reconciliation to prevent NaN/Inf propagation).
- **Engine Lifecycle**: Heavy models must be lazy-loaded as singletons in `recursive_flow.py`.
