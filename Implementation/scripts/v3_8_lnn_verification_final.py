import sys
import os
import logging
import numpy as np

# Add the project path to sys.path
PROJECT_PATH = r"d:\Antigravity\Test VJEPA EVENTBASED LLM\Implementation"
LNN_REPO_PATH = r"d:\Antigravity\Test VJEPA EVENTBASED LLM\Techs\LNN-master\LNN-master"

if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)
if LNN_REPO_PATH not in sys.path:
    sys.path.append(LNN_REPO_PATH)

# Configure logging
logging.basicConfig(level=logging.ERROR)
from v2_logic.models.lnn_knowledge_base import get_lnn_kb


def test_lnn_intent_filter():
    print("\n" + "=" * 50)
    print("--- FINAL TESTING LNN INTENT FILTER ---")
    print("=" * 50)
    kb = get_lnn_kb()

    test_cases = [
        ("Ayam Brand Beans", True, "Specific Product"),
        ("know", False, "Procedural Noise"),
        ("fact", False, "Procedural Noise"),
        ("Amoy Corn", True, "Specific Product (New Category)"),
        ("please count", False, "Conversational Noise"),
    ]

    all_passed = True
    for text, should_pass, desc in test_cases:
        score = kb.validate_intent(text)
        # Use lower thresholds for strictness test - Valid if > 0.6
        status = "PASSED" if (score > 0.6) == should_pass else "FAILED"
        if status == "FAILED":
            all_passed = False
        print(f"[{status}] '{text:20}' | Score: {score:.4f} | {desc}")
    return all_passed


def test_lnn_identity_reconciliation():
    print("\n" + "=" * 50)
    print("--- FINAL TESTING LNN IDENTITY RECONCILIATION ---")
    print("=" * 50)
    kb = get_lnn_kb()

    # Use UNIQUE IDs to avoid fact collision in singleton LNN Model
    test_cases = [
        ("final_id_1", "Ayam Brand Beans", 0.95, 1.0, True, "Perfect Match"),
        (
            "final_id_2",
            "Stock Cube",
            0.95,
            0.05,
            False,
            "Visual Match but Physical Impossible",
        ),
        ("final_id_3", "Amoy Corn", 0.1, 1.0, False, "Consistent but Visual Mismatch"),
        (
            "final_id_4",
            "Kaleng Kristal",
            0.8,
            1.0,
            True,
            "Strong Match + Consistent Volume",
        ),
    ]

    all_passed = True
    for cid, sku, sim, vol_r, should_pass, desc in test_cases:
        score = kb.reconcile_identity(cid, sku, sim, vol_r)
        # Since we use Lower Bound, anything > 0.5 is a confident match
        status = "PASSED" if (score > 0.5) == should_pass else "FAILED"
        if status == "FAILED":
            all_passed = False
        print(f"[{status}] Identity({cid:10}, {sku:15}) | Score: {score:.4f} | {desc}")
    return all_passed


if __name__ == "__main__":
    t1 = test_lnn_intent_filter()
    t2 = test_lnn_identity_reconciliation()

    print("\n" + "=" * 50)
    if t1 and t2:
        print("RESULT: ALL NEURO-SYMBOLIC LNN TESTS PASSED [READY FOR PRODUCTION]")
    else:
        print("RESULT: VERIFICATION FAILED - SYSTEM REJECTED")
    print("=" * 50)
