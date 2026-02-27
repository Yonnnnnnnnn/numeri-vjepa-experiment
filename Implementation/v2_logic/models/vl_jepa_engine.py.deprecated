"""
VL-JEPA Engine (Director)

Integrates PaliGemma weights into a JEPA-style intent identifier.
Note: Since the source Techs/VL-JEPA uses MLX (Apple Silicon only),
this implementation uses PyTorch/Transformers for compatibility on Windows.

Pattern: Proxy / Facade
- Proxies the PaliGemma VLM to act as a JEPA semantic director.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : VLJEPAEngine (this module)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <VLJEPAEngine> → class implementation                                    │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <PaliGemma>    ← from transformers                                       │
  │  <AutoProcessor> ← from transformers                                       │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : str, float, bool, "cuda", "cpu"

Production Rules:
  VLJEPAEngine    → imports + <VLJEPAEngine>
  <VLJEPAEngine>  → __init__ + identify_intent + identify_foveated_intents
                   + extract_visual_embeddings + _frame_to_pil
═══════════════════════════════════════════════════════════════════════════════
"""

import re

import numpy as np
import torch
import logging
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

logger = logging.getLogger(__name__)


class VLJEPAEngine:
    """
    Director engine for autonomous intent and identification.
    Uses PaliGemma weights to map visual inputs to semantic SKUs.

    Pattern: Facade
    """

    def __init__(
        self, model_id="google/paligemma-3b-mix-224", device="cuda", token=None
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.token = token

        logger.info(f"[VL-JEPA] Loading model: {model_id} on {self.device}")

        # Set memory configuration for CUDA
        if self.device == "cuda":
            import os

            os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        # Load PaliGemma with token authentication
        self.processor = AutoProcessor.from_pretrained(model_id, token=self.token)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            low_cpu_mem_usage=True,
            offload_folder="./offload",
            offload_state_dict=True,
            token=self.token,
            # Use eager attention to avoid SDPA mask_function issues on torch<2.6
            attn_implementation="eager",
        ).eval()

        # Clear cache after loading
        if self.device == "cuda":
            torch.cuda.empty_cache()
            import gc

            gc.collect()

        logger.info("[VL-JEPA] Model loaded successfully")

    def _frame_to_pil(self, frame) -> Image.Image:
        """Convert frame (Tensor, ndarray, str, or PIL.Image) to PIL.Image."""
        if isinstance(frame, torch.Tensor):
            if frame.shape[0] == 3:  # C, H, W -> H, W, C
                frame = frame.permute(1, 2, 0)
            return Image.fromarray((frame.cpu().numpy() * 255).astype("uint8"))
        if isinstance(frame, str):
            return Image.open(frame)
        if isinstance(frame, np.ndarray):
            return Image.fromarray(frame)
        return frame  # Already PIL

    def identify_intent(
        self,
        frame,
        prompt="What is the main object type in this image? Answer with a single word.",
        default_intent="items",
    ):
        """
        Identify the scanning context using vision-language reasoning.
        This is the PERIPHERAL (wide-angle) identification mode.

        Args:
            frame: np.ndarray, PIL.Image or path (H, W, 3)
            prompt: str, the query to ask the director.
            default_intent: str, fallback intent if identification fails.

        Returns:
            str: The identified intent/SKU name.
        """
        img = self._frame_to_pil(frame)

        inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,  # Lower temperature for more consistent results
                top_p=0.9,  # Nucleus sampling for better quality
                do_sample=False,  # Deterministic generation
            )

        # Output logic
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # Clean the output
        intent = output_text[len(prompt) :].strip().lower()

        # Handle cases where output is empty or not useful
        if (
            not intent
            or len(intent) > 20
            or any(char in intent for char in [",", ".", ";", "!", "?"])
        ):
            # Extract first word if multiple words are returned
            intent_words = intent.split()
            if intent_words:
                intent = intent_words[0]
            else:
                intent = default_intent

        logger.info("[VL-JEPA] Identified Intent: %s", intent)
        return intent

    def identify_foveated_intents(
        self,
        cropped_frame,
        user_prompt: str = "",
        default_intents: list = None,
    ) -> list:
        """
        FOVEATED (close-up) multi-SKU identification on a PointBeam-cropped image.

        Unlike identify_intent which forces a single-word answer on a wide
        frame (causing Intent Collapse), this method:
        1. Accepts a PointBeam-cropped (zoomed) image for higher detail.
        2. Uses a descriptive prompt that asks for MULTIPLE distinct labels.
        3. Parses the VLM response into a list of individual SKU intents.

        Args:
            cropped_frame: np.ndarray or PIL.Image — PointBeam-cropped region.
            user_prompt: Original user prompt for context (e.g. "count cans by brand").
            default_intents: Fallback list if VLM returns nothing useful.

        Returns:
            List[str]: Distinct intent labels discovered in the foveated view.
        """
        if default_intents is None:
            default_intents = ["items"]

        img = self._frame_to_pil(cropped_frame)

        # Build a context-aware foveated prompt
        if user_prompt:
            foveated_prompt = (
                f"Context: {user_prompt}. "
                "Look at this close-up image carefully. "
                "List all distinct object types or brand labels you can see, "
                "separated by commas. Be specific with brand names."
            )
        else:
            foveated_prompt = (
                "Look at this close-up image carefully. "
                "List all distinct object types or brand labels you can see, "
                "separated by commas. Be specific."
            )

        inputs = self.processor(
            text=foveated_prompt, images=img, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=100,  # Allow longer response for multi-SKU
                temperature=0.3,  # Slightly creative for brand discovery
                top_p=0.9,
                do_sample=True,
            )

        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # Extract the generated part (after the prompt)
        raw_response = output_text[len(foveated_prompt) :].strip().lower()
        logger.info("[VL-JEPA] Foveated raw response: '%s'", raw_response)

        # Parse comma/newline/semicolon separated labels
        # Also handle "and" as a separator
        raw_response = raw_response.replace(" and ", ",")
        raw_response = raw_response.replace(";", ",")
        raw_response = raw_response.replace("\n", ",")

        # Split and clean
        candidates = [s.strip() for s in raw_response.split(",")]

        # Filter out empty strings, numbering artifacts, and overly long items
        intents = []
        for candidate in candidates:
            # Remove leading numbering like "1. " or "- "
            cleaned = re.sub(r"^[\d\-\.\)\]]+\s*", "", candidate).strip()
            if not cleaned:
                continue
            if len(cleaned) > 40:  # Skip overly verbose descriptions
                continue
            if cleaned not in intents:  # Deduplicate
                intents.append(cleaned)

        if not intents:
            logger.warning(
                "[VL-JEPA] Foveated identification returned no valid intents. "
                "Using defaults: %s",
                default_intents,
            )
            return list(default_intents)

        logger.info(
            "[VL-JEPA] Foveated Multi-SKU Discovery: %d intents found: %s",
            len(intents),
            intents,
        )
        return intents

    def extract_visual_embeddings(self, frame):
        """
        Extract visual latent features for the JEPA world model.

        Returns:
            torch.Tensor: High-dimensional visual features.
        """
        # We can extract the 'vision_tower' hidden states which is equivalent
        # to the X-Encoder in the VL-JEPA paper.
        img = self._frame_to_pil(frame)

        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            vision_outputs = self.model.vision_tower(inputs.pixel_values)
            # PaliGemma vision tower usually outputs (B, L, D)
            features = vision_outputs.last_hidden_state

        return features
