"""
Targeted SLM Engine (Reasoning Core)

Wraps the VLM to provide high-level reasoning for anomalies detected by the Logic Gate.
Acts as the "Cognitive System 2" that is only triggered when "System 1" (Perception) fails.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : SLMEngineModule (this module)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <SLMEngine>      → Main reasoning engine                                 │
  │  <ReasoningResult> → Output dataclass with explanation and hypothesis     │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <VLMInferenceModel> ← from models.vlm_wrapper (Base VLM)                 │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : str, dict, Image, int

Production Rules:
  SLMEngineModule → imports + <ReasoningResult> + <SLMEngine>
  SLMEngine       → __init__ + generate_reasoning + _construct_prompt
                  + estimate_object_volume + generate_initial_intents
═══════════════════════════════════════════════════════════════════════════════

Pattern: Facade (wrapping VLM) + Strategy (different prompts for different anomalies)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReasoningResult:
    """Structured output from SLM reasoning."""

    reasoning_text: str  # The raw explanation
    hypothesis: str  # Short hypothesis for next action
    confidence: float  # Estimated confidence of reasoning


class SLMEngine:
    """
    Targeted Small Language Model (SLM) Engine.

    Role: Analyze anomalies flagged by LogicGate and generate hypotheses.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2-VL-7B-Instruct"):
        """
        Initialize the SLM Engine.
        Lazy-loads the heavy VLM model only when first needed (handled by lazy loader in controller),
        but here we assume instance creation means we want the model.
        """
        self.vlm = None
        self.model_id = model_id

    def _ensure_model_loaded(self):
        """Load VLM if not already loaded."""
        if self.vlm is None:
            try:
                from .vlm_wrapper import VLMInferenceModel

                logger.info("[SLMEngine] Loading VLM backend...")
                self.vlm = VLMInferenceModel(model_id=self.model_id)
            except ImportError as e:
                logger.error("[SLMEngine] Failed to import VLM wrapper: %s", e)
                raise
            except Exception as e:
                logger.error("[SLMEngine] Failed to load VLM: %s", e)
                raise

    def generate_reasoning(
        self,
        image: np.ndarray,
        anomaly_type: str,
        context: Dict[str, Any],
    ) -> ReasoningResult:
        """
        Generate reasoning for a specific anomaly.

        Args:
            image: RGB image of the scene.
            anomaly_type: 'spatial', 'volumetric', or 'confidence'.
            context: Context details (e.g. residue ratio, current count).

        Returns:
            ReasoningResult with explanation and hypothesis.
        """
        self._ensure_model_loaded()

        # Construct prompted based on anomaly type
        prompt = self._construct_prompt(anomaly_type, context)
        logger.info("[SLMEngine] Prompt: %s", prompt)

        # Run inference
        try:
            response_text = self.vlm.predict(image, prompt_text=prompt)
            logger.info("[SLMEngine] Response: %s", response_text)

            # Parse response (Simple heuristic parsing for now)
            # In Phase 3, we can use structured generation/JSON mode
            reasoning = response_text
            hypothesis = self._extract_hypothesis(response_text)

            return ReasoningResult(
                reasoning_text=reasoning,
                hypothesis=hypothesis,
                confidence=0.7,  # Placeholder confidence
            )

        except Exception as e:
            logger.error("[SLMEngine] Inference failed: %s", e)
            return ReasoningResult(
                reasoning_text="Error during SLM inference.",
                hypothesis="Retry logic.",
                confidence=0.0,
            )

    def _construct_prompt(self, anomaly_type: str, context: Dict[str, Any]) -> str:
        """Construct context-aware prompt."""
        base_prompt = "You are an intelligent visual analyst. "

        if anomaly_type == "spatial":
            residue = context.get("residue_ratio", 0.0)
            base_prompt += (
                f"I detected {residue:.1%} unexplained motion/energy in the scene "
                "that was NOT covered by the main object masks. "
                "Look at the image carefully. Is there an object moving that was missed? "
                "Or is it just shadow/noise? Explain what you see in the background."
            )

        elif anomaly_type == "volumetric":
            n_visible = context.get("n_visible", 0)
            vol_range = context.get("n_volumetric_range", (0, 0))
            base_prompt += (
                f"I counted {n_visible} objects visually, but the 3D volume suggests "
                f"there should be between {vol_range[0]} and {vol_range[1]} objects. "
                "This is a discrepancy. Are some objects occluded (hidden behind others)? "
                "Or are some counts false positives? Analyze the spatial arrangement."
            )

        else:
            base_prompt += (
                "I am unsure about the current count. "
                "Analyze the image and tell me if there are any ambiguous objects, "
                "occlusions, or lighting issues affecting visibility."
            )

        base_prompt += "\nProvide a concise explanation and a hypothesis."
        return base_prompt

    def _extract_hypothesis(self, text: str) -> str:
        """Extract short hypothesis from text (heuristic)."""
        # Simple extraction: First sentence or key phrase
        sentences = text.split(".")
        if len(sentences) > 0:
            return sentences[0].strip()
        return "Unknown anomaly."

    def _extract_generic_anchor(self, label: str) -> str:
        """
        Semantic Pivot: Extracts a generic noun from a specific label.
        Example: "Blue Plastic Cup" -> "cup"
        """
        # Daftar anchor words yang umum diketahui VLM memiliki volume standar
        # Diurutkan dari yang spesifik ke umum (Generic Anchor Strategy)
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

        # 1. Exact match check (jika label sudah generik)
        if label_lower in anchors:
            return label_lower

        # 2. Substring match (prioritize longer matches if overlap, or order in list)
        for anchor in anchors:
            # Menggunakan regex boundary (\b) agar tidak match parsial (misal 'can' di 'candy')
            # tapi simple substring check seringkali cukup untuk MVP
            if anchor in label_lower:
                return anchor

        # 3. Fallback: Return original label
        return label

    def estimate_object_volume(
        self, label: str, image: Optional[np.ndarray] = None
    ) -> float:
        """
        Ask SLM for a physical volume estimate (prior) for a given object label.

        Args:
            label: Object label (e.g., "bottle", "person").
            image: Context image (optional).

        Returns:
            Estimated volume in m^3 (float).
        """
        self._ensure_model_loaded()

        # --- SEMANTIC PIVOT ---
        # Gunakan label generik untuk pertanyaan fisika ke VLM agar tidak bingung
        query_label = self._extract_generic_anchor(label)

        if query_label != label:
            logger.info(
                "[SLMEngine] Semantic Pivot: '%s' -> '%s' for volume estimation",
                label,
                query_label,
            )

        prompt = (
            f"Estimate the average physical volume of a single '{query_label}' in cubic meters (m^3). "
            "Consider standard real-world dimensions for this object. "
            "Provide ONLY the numeric value in scientific notation (e.g., 5.0e-4) or simple decimal (e.g., 0.0005). "
            "Absolutely NO units, NO explanation, NO symbols other than numbers and decimal/scientific notation."
        )

        # Use a blank image if none provided (pure knowledge retrieval)
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        try:
            logger.info("[SLMEngine] Requesting volume prior for '%s'", label)
            # Use very low max_new_tokens to prevent rambling/zeros
            response = self.vlm.predict(image, prompt_text=prompt, max_new_tokens=24)
            logger.info("[SLMEngine] Volume Response: %s", response)

            # Parse number
            import re

            # CLEANUP: Remove any non-numeric noise before regex
            clean_resp = "".join(c for c in response if c in "0123456789.eE-+")
            match = re.search(r"[-+]?\d*\.?\d+([eE][-+]?\d+)?", clean_resp)

            # Context-aware fallback values (in m^3)
            fallbacks = {
                "cup": 0.00025,
                "bottle": 0.0005,
                "ball": 0.00005,
                "can": 0.00033,
            }
            default_fallback = 0.001

            if match:
                val = float(match.group())
                # Rule 6.1/6.10: Safety Floor (1cm^3) to prevent division by zero or extreme counts
                if val < 1e-6:
                    logger.warning(
                        "[SLMEngine] Volume too low (%.2e), checking context fallback",
                        val,
                    )
                    # V3.3.2 Hotfix: Use context-aware fallback BEFORE safety floor
                    for k, v in fallbacks.items():
                        if k in label.lower():
                            logger.info(
                                "[SLMEngine] Context fallback for '%s': %.6f m^3",
                                label,
                                v,
                            )
                            return v
                    # Generic fallback (1 liter) — still better than 1e-6
                    return default_fallback

                # Rule 6.9: Sanity bounds (filters out hallucinations like 1000m^3 for a bottle)
                if val > 1.0:  # Nothing we count is bigger than 1 cubic meter
                    logger.warning(
                        "[SLMEngine] Volume too high (%.2f), clamping to 1.0", val
                    )
                    return 1.0

                return val
            else:
                logger.warning("[SLMEngine] Could not parse volume from: %s", response)
                # Use contextual fallback if label matches common objects
                for k, v in fallbacks.items():
                    if k in label.lower():
                        return v
                return default_fallback

        except Exception as e:
            logger.error("[SLMEngine] Volume estimation failed: %s", e)
            return 0.001

    def generate_initial_intents(
        self,
        prompt: str,
        frame: np.ndarray,
        detections: List[Any],
    ) -> List[Dict[str, Any]]:
        """
        V3.3 Step 0 Analyst: Generate specific SKU names from user prompt + detections.

        Uses VLM to analyze GroundingDINO detections and produce specific labels
        (e.g., 'Coca Cola 330ml Bottle' instead of generic 'bottle').

        Args:
            prompt: Raw user prompt (e.g., 'count soda').
            frame: The best frame selected by GroundingDINO scout.
            detections: Raw detections from GroundingDINO.

        Returns:
            List of genesis intent dicts:
            [{'label': str, 'confidence': float, 'source': 'vlm'}]
        """
        self._ensure_model_loaded()

        n_detected = len(detections) if detections else 0

        vlm_prompt = (
            f"The user's EXPLICIT instruction is: '{prompt}'.\n"
            f"I detected {n_detected} candidate objects.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. MANDATORY: You MUST identify EVERY unique type of object requested in the prompt (PRO objects).\n"
            "2. Even if an object is small or partially occluded, you MUST provide an INTENT line for it if it matches the prompt.\n"
            "3. Identify SPECIFIC product names (SKUs) based on visual features.\n"
            "4. Identify UNRELATED objects (human hands, table, background) as 'CONTRA'.\n\n"
            "Reply in this EXACT format (one per line):\n"
            "INTENT: [specific SKU label]\n"
            "CONTRA: [distractor label]\n\n"
            "Example prompt: 'count soda and chips'\n"
            "INTENT: Coca Cola 330ml Can\n"
            "INTENT: Pringles Sour Cream Canister\n"
            "CONTRA: human hand\n"
        )

        try:
            response = self.vlm.predict(
                frame, prompt_text=vlm_prompt, max_new_tokens=64
            )
            logger.info("[SLMEngine] Genesis response: %s", response)

            intents = []
            for line in response.strip().split("\n"):
                line = line.strip()
                if line.upper().startswith("INTENT:"):
                    label = line[len("INTENT:") :].strip()
                    if label:
                        intents.append(
                            {
                                "label": label,
                                "confidence": 0.8,
                                "source": "vlm",
                                "is_contra": False,
                            }
                        )
                elif line.upper().startswith("CONTRA:"):
                    label = line[len("CONTRA:") :].strip()
                    if label:
                        intents.append(
                            {
                                "label": label,
                                "confidence": 0.7,
                                "source": "vlm",
                                "is_contra": True,
                            }
                        )

            # --- Safety Validation (Defense in Depth) ---
            validated_intents = []

            # Refined keyword extraction
            stop_words = {
                "count",
                "the",
                "and",
                "with",
                "there",
                "are",
                "many",
                "please",
                "video",
            }
            user_keywords = [
                k.lower().strip(",").strip().rstrip("s")
                for k in prompt.lower().split()
                if len(k) > 2 and k not in stop_words
            ]

            found_keywords = set()
            for item in intents:
                label_lower = item["label"].lower()

                # Semantic Guard: PRO Verification
                is_pro = False
                for kw in user_keywords:
                    if kw in label_lower or label_lower in kw:
                        is_pro = True
                        found_keywords.add(kw)
                        break

                if is_pro and item["is_contra"]:
                    logger.warning(
                        "[SLMEngine] PRO GUARD: '%s' found in CONTRA but matches PRO keyword. Overriding to INTENT.",
                        item["label"],
                    )
                    item["is_contra"] = False

                # Safety Override for Hands
                if (
                    not item["is_contra"]
                    and "hand" in label_lower
                    and "hand" not in user_keywords
                ):
                    item["is_contra"] = True

                validated_intents.append(item)

            # Check for MISSING PRO keywords (Mandatory Object Verification)
            for kw in user_keywords:
                if kw not in found_keywords:
                    logger.warning(
                        "[SLMEngine] MISSING PRO: keyword '%s' from prompt not found in VLM intents. Force adding generic intent.",
                        kw,
                    )
                    validated_intents.append(
                        {
                            "label": kw.capitalize(),
                            "confidence": 0.4,
                            "source": "pro_guard_fallback",
                            "is_contra": False,
                        }
                    )

            if not validated_intents:
                # VLM did not follow format — fallback to prompt
                logger.warning("[SLMEngine] Could not parse intents, using fallback")
                validated_intents = [
                    {
                        "label": prompt,
                        "confidence": 0.5,
                        "source": "fallback",
                        "is_contra": False,
                    }
                ]

            return validated_intents

        except Exception as e:
            logger.error("[SLMEngine] generate_initial_intents failed: %s", e)
            return [
                {
                    "label": prompt,
                    "confidence": 0.3,
                    "source": "error",
                    "is_contra": False,
                }
            ]
