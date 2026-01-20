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
- **Naming**: PEP 8 (snake_case for functions/variables, PascalCase for classes)

## Enforced Rules

- **Rule 2.1**: Law of Flatness (Guard Clauses first)
- **Rule 2.2**: Law of DRY (Shared utilities for vision filters)
- **Rule 3**: GoF Patterns (Adapter, Facade, Strategy)
- **Rule 7**: CFG Documentation in every code file
