import sys


# Mock class with updated logic
class MockSLMEngine:
    def _extract_generic_anchor(self, label: str) -> str:
        """
        Semantic Pivot: Extracts a generic noun from a specific label.
        Example: "Blue Plastic Cup" -> "cup"
        """
        # Matches the implementation in slm_engine.py (Reordered)
        anchors = [
            # Specific containers matched first
            "mug",
            "tumbler",
            "jar",
            "jug",
            "cup",
            "bottle",
            "can",
            # Ambiguous materials/shapes processed later
            "glass",
            "ball",
            "sphere",
            # Boxes & Packaging
            "carton",
            "box",
            "container",
            "pouch",
            "sachet",
            "bag",
            # Tools
            "marker",
            "pencil",
            "pen",
            # Tech (Smartphone before phone)
            "smartphone",
            "laptop",
            "phone",
            # Furniture
            "chair",
            "table",
            # People
            "woman",
            "man",
            "person",
            "human",
        ]

        label_lower = label.lower()

        if label_lower in anchors:
            return label_lower

        for anchor in anchors:
            # Using simple substring match as in implementation
            if anchor in label_lower:
                return anchor

        return label


def test_semantic_pivot():
    engine = MockSLMEngine()

    test_cases = [
        ("Blue Plastic Cup", "cup"),
        ("Coca Cola Can", "can"),
        ("Red Ball", "ball"),
        ("My Water Bottle", "bottle"),
        ("Unknown Gizmo", "Unknown Gizmo"),
        ("Glass Jar", "jar"),  # Should now be 'jar' not 'glass'
        ("Paper Bag", "bag"),
        ("Wooden Chair", "chair"),
        ("A large metallic sphere", "sphere"),
        ("iphone 15 pro max", "phone"),
        ("samsung smartphone", "smartphone"),  # Should now be 'smartphone' not 'phone'
    ]

    print("\n--- Testing Semantic Pivot Logic (Reordered) ---")
    passed = 0
    for specific, expected in test_cases:
        result = engine._extract_generic_anchor(specific)
        status = "✅" if result == expected else "❌"
        print(
            f"{status} Input: '{specific}' -> Output: '{result}' (Expected: '{expected}')"
        )
        if result == expected:
            passed += 1
        else:
            print(f"   [FAILED] Expected '{expected}', got '{result}'")

    print(f"\nResult: {passed}/{len(test_cases)} passed.")

    if passed == len(test_cases):
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")


if __name__ == "__main__":
    test_semantic_pivot()
