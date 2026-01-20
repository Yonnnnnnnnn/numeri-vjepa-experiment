"""
Fusion Engine Module

Combines VLM visual understanding with V-JEPA temporal memory
to produce accurate inventory counts including occluded items.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : FusionEngine (this file)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <FusionEngine>    → Main fusion logic                                    │
  │  <FusedResult>     → Output dataclass                                     │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <TemporalContext>  ← from v2_logic.models.temporal_memory (occlusion info)│
  │  <VLMResult>        ← from vlm_wrapper (visible count + description)      │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : int, float, str

Production Rules:
  FusionEngine → fuse(vlm_result, temporal_context) → FusedResult
═══════════════════════════════════════════════════════════════════════════════

Pattern: Strategy
- Encapsulates the fusion algorithm
- Can be swapped for different fusion strategies (simple, weighted, ML-based)
"""

import re
from dataclasses import dataclass
from typing import Optional

from v2_logic.models.temporal_memory import TemporalContext


@dataclass
class FusedResult:
    """Final fused output combining VLM and temporal memory."""

    total_count: int  # Total items (visible + occluded)
    visible_count: int  # Items currently visible (from VLM)
    occluded_count: int  # Items tracked from memory
    description: str  # VLM's text description
    summary: str  # Human-readable summary
    confidence: float  # Overall confidence (0-1)


class FusionEngine:
    """
    Fuses VLM output with V-JEPA temporal memory.

    Combines:
    - VLM's visible count and description
    - Temporal memory's occluded item estimate
    """

    def __init__(self, occlusion_weight: float = 0.8):
        """
        Args:
            occlusion_weight: How much to trust occlusion estimates (0-1)
        """
        self.occlusion_weight = occlusion_weight

    @staticmethod
    def parse_count_from_text(text: str) -> int:
        """
        Extract numeric count from VLM text response.

        Handles various formats:
        - "I can see 12 items"
        - "There are approximately 15 bottles"
        - "Count: 20"
        - "12"
        """
        if not text:
            return 0

        # Try to find explicit count patterns
        patterns = [
            r"(\d+)\s*(?:items?|objects?|products?|bottles?|boxes?|packages?)",
            r"(?:count|total|see|are|found)\s*(?:is|:)?\s*(\d+)",
            r"approximately\s*(\d+)",
            r"^(\d+)$",  # Just a number
            r"(\d+)",  # Any number as fallback
        ]

        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))

        return 0

    def fuse(
        self,
        vlm_text: str,
        temporal_context: Optional[TemporalContext] = None,
    ) -> FusedResult:
        """
        Fuse VLM output with temporal context.

        Args:
            vlm_text: Raw text output from VLM
            temporal_context: Temporal memory analysis (optional)

        Returns:
            FusedResult with combined analysis
        """
        # Parse VLM count
        visible_count = self.parse_count_from_text(vlm_text)

        # Handle case without temporal context
        if temporal_context is None:
            return FusedResult(
                total_count=visible_count,
                visible_count=visible_count,
                occluded_count=0,
                description=vlm_text,
                summary=f"{visible_count} items visible",
                confidence=0.5,
            )

        # Calculate weighted occlusion estimate
        raw_occluded = temporal_context.occluded_estimate
        weighted_occluded = int(
            raw_occluded * self.occlusion_weight * temporal_context.confidence
        )

        # Total count
        total = visible_count + weighted_occluded

        # Override with current visible if it's higher than our estimate
        # (VLM might see more than we tracked)
        total = max(total, visible_count)

        # Build summary
        if weighted_occluded > 0:
            summary = (
                f"{total} items total "
                f"({visible_count} visible + {weighted_occluded} from memory)"
            )
        else:
            summary = f"{visible_count} items visible"

        # Overall confidence
        confidence = min(
            1.0,
            (0.7 + 0.3 * temporal_context.confidence)
            * (0.5 + 0.5 * temporal_context.feature_similarity),
        )

        return FusedResult(
            total_count=total,
            visible_count=visible_count,
            occluded_count=weighted_occluded,
            description=vlm_text,
            summary=summary,
            confidence=confidence,
        )


if __name__ == "__main__":
    # Quick test
    engine = FusionEngine()

    # Test count parsing
    test_texts = [
        "I can see 12 bottles on the shelf",
        "There are approximately 15 items",
        "Count: 20",
        "25",
        "The image shows about 8 boxes and 5 bottles",
    ]

    for text_case in test_texts:
        count_val = engine.parse_count_from_text(text_case)
        print(f"'{text_case}' → {count_val}")

    # Test fusion
    # (TemporalContext is already imported at top)

    ctx = TemporalContext(
        peak_count=15,
        current_visible=10,
        occluded_estimate=5,
        confidence=0.9,
        feature_similarity=0.85,
    )

    result = engine.fuse("I see 10 items on the shelf", ctx)
    print(f"\nFused result: {result}")
