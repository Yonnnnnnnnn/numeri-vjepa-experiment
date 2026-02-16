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

        prompt = (
            f"Estimate the average physical volume of a '{label}' in cubic meters (m^3). "
            "Consider standard real-world dimensions. "
            "Reply with a SINGLE numeric value in scientific notation or decimal. "
            "Example: 0.0005. Do NOT include units or explanation."
        )

        # Use a blank image if none provided (pure knowledge retrieval)
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        try:
            logger.info("[SLMEngine] Requesting volume prior for '%s'", label)
            response = self.vlm.predict(image, prompt_text=prompt)
            logger.info("[SLMEngine] Volume Response: %s", response)

            # Parse number
            import re

            # Match scientific notation or float: 1.2e-3, 0.005, 5e-4
            match = re.search(r"[-+]?\d*\.?\d+([eE][-+]?\d+)?", response)
            if match:
                val = float(match.group())
                # Sanity bounds (filters out hallucinations like 1000m^3 for a bottle)
                if val <= 0:
                    return 0.001
                return val
            else:
                logger.warning("[SLMEngine] Could not parse volume from: %s", response)
                return 0.001  # Fallback

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
            f"The user asked to '{prompt}'. I detected {n_detected} candidate objects "
            "in this image using a visual detector.\n\n"
            "Your task:\n"
            "1. Look at the detected regions and identify the SPECIFIC product/object names.\n"
            "2. Distinguish between different types if there are multiple object categories.\n"
            "3. Identify any objects that are NOT relevant to the user's query "
            "(e.g., hands, shadows, background clutter).\n\n"
            "Reply in this EXACT format (one per line):\n"
            "INTENT: [specific label]\n"
            "CONTRA: [distractor label]\n\n"
            "Example:\n"
            "INTENT: Coca Cola 330ml Bottle\n"
            "INTENT: Sprite 500ml Bottle\n"
            "CONTRA: human hand\n"
        )

        try:
            response = self.vlm.predict(frame, prompt_text=vlm_prompt)
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

            if not intents:
                # VLM did not follow format — fallback to prompt
                logger.warning("[SLMEngine] Could not parse intents, using fallback")
                intents = [
                    {
                        "label": prompt,
                        "confidence": 0.5,
                        "source": "fallback",
                        "is_contra": False,
                    }
                ]

            return intents

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
