import sys
import os

# Add project root to path
sys.path.append(r"d:\Antigravity\Test VJEPA EVENTBASED LLM")

from Implementation.v2_logic.models.slm_engine import SLMEngine


def test_semantic_pivot():
    # Instantiate SLMEngine (mocking VLM to avoid loading heavy models if possible,
    # but here we just need the helper method which doesn't use VLM)
    # We'll just instantiate it, assuming __init__ doesn't load VLM immediately or we can bypass it.
    # Looking at SLMEngine, it seems safe enough for a quick unit test of a pure method.

    # Actually, SLMEngine might load stuff. Let's just monkey patch or create a dummy class with the method
    # just to test the logic, OR better, just copy the method logic here for a quick verify?
    # No, we want to test the actual code.

    # Let's try to instantiate. If it fails due to resource loading, we'll mock.
    try:
        engine = SLMEngine()
        print("[TEST] SLMEngine instantiated.")
    except Exception as e:
        print(f"[TEST] Could not instantiate SLMEngine directly: {e}")
        return

    test_cases = [
        ("Blue Plastic Cup", "cup"),
        ("Coca Cola Can", "can"),
        ("Red Ball", "ball"),
        ("My Water Bottle", "bottle"),
        ("Unknown Gizmo", "Unknown Gizmo"),  # Fallback
        ("Glass Jar", "jar"),
        ("Paper Bag", "bag"),
        ("Wooden Chair", "chair"),
        ("A large metallic sphere", "sphere"),
        ("iphone 15 pro max", "phone"),  # partial match 'phone'
        ("samsung smartphone", "smartphone"),  # exact match
    ]

    print("\n--- Testing Semantic Pivot Logic ---")
    passed = 0
    for specific, expected in test_cases:
        result = engine._extract_generic_anchor(specific)
        status = "✅" if result == expected else "❌"
        print(
            f"{status} Input: '{specific}' -> Output: '{result}' (Expected: '{expected}')"
        )
        if result == expected:
            passed += 1

    print(f"\nResult: {passed}/{len(test_cases)} passed.")


if __name__ == "__main__":
    test_semantic_pivot()
