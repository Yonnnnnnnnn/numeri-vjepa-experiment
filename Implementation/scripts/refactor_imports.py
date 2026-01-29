"""
refactor_imports.py

Utility script.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : Script (this file)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <Main>           → Script entry point                                    │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <Imports>        ← External dependencies                                 │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : str, int, etc.

Production Rules:
  Script          → Imports + Main
═══════════════════════════════════════════════════════════════════════════════

Pattern: Script
- Standalone executable for utility/testing purposes.

"""
import os

ROOT_DIR = r"d:\Antigravity\Test VJEPA EVENTBASED LLM\Implementation\vjepa_src"


def refactor_files():
    print(f"Refactoring imports in {ROOT_DIR}...")
    for root, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Replace rules
                # 1. from src. -> from vjepa_src.
                # 2. from src import -> from vjepa_src import
                # 3. import src. -> import vjepa_src.

                new_content = content.replace("from src.", "from vjepa_src.")
                new_content = new_content.replace("from src ", "from vjepa_src ")
                new_content = new_content.replace("import src.", "import vjepa_src.")

                if new_content != content:
                    print(f"Updating: {path}")
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(new_content)


if __name__ == "__main__":
    refactor_files()
