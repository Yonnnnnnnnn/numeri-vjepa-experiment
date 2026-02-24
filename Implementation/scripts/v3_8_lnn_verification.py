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

# Configure logging to be less noisy for tests
logging.basicConfig(level=logging.INFO)
# Silence LNN unless error
logging.getLogger("lnn").setLevel(logging.ERROR)

from v2_logic.models.lnn_knowledge_base import get_lnn_kb


def test_lnn_intent_filter():
    print("\n" + "=" * 50)
    print("--- TESTING LNN INTENT FILTER (V3.8) ---")
    print("=" * 50)
    kb = get_lnn_kb()

    test_cases = [
        ("Ayam Brand Baked Beans", True),
        ("know", False),
        ("fact", False),
        ("task", False),
        ("Amoy Corn", True),
        ("please identify labels", False),
    ]

    success = True
    for text, should_be_valid in test_cases:
        score = kb.validate_intent(text)
        status = "PASSED" if (score > 0.6) == should_be_valid else "FAILED"
        if status == "FAILED":
            success = False
        print(
            f"[{status}] Text: '{text:25}' | Score: {score:.4f} | Should be valid: {should_be_valid}"
        )
    return success


def test_lnn_identity_reconciliation():
    print("\n" + "=" * 50)
    print("--- TESTING LNN IDENTITY RECONCILIATION ---")
    print("=" * 50)
    kb = get_lnn_kb()

    # cluster_id, sku_label, similarity, vol_ratio
    test_cases = [
        ("id_1", "Ayam Brand Beans", 0.9, 1.0, True, "Match + Consistent"),
        ("id_2", "Stock Cube", 0.9, 0.1, False, "Match but Impossible Volume"),
        ("id_3", "Amoy Corn", 0.3, 1.0, False, "Consistent but No Visual Match"),
        (
            "id_4",
            "Ayam Brand Beans",
            0.8,
            0.2,
            False,
            "High Match but Cluster too small for object",
        ),
    ]

    success = True
    for cid, sku, sim, vol_r, should_pass, desc in test_cases:
        score = kb.reconcile_identity(cid, sku, sim, vol_r)
        status = "PASSED" if (score > 0.5) == should_pass else "FAILED"
        if status == "FAILED":
            success = False
        print(
            f"[{status}] Identity({cid}, {sku}) | Sim: {sim}, VolR: {vol_r} | Score: {score:.4f} | {desc}"
        )
    return success


if __name__ == "__main__":
    s1 = test_lnn_intent_filter()
    s2 = test_lnn_identity_reconciliation()

    print("\n" + "=" * 50)
    if s1 and s2:
        print("RESULT: ALL NEURO-SYMBOLIC TESTS PASSED [OK]")
    else:
        print("RESULT: SOME TESTS FAILED [ERR]")
    print("=" * 50)
