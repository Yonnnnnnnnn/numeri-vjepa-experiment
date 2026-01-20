"""
VL-JEPA Engine (Director)

Integrates PaliGemma weights into a JEPA-style intent identified.
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
  <VLJEPAEngine>  → __init__ + identify_intent + extract_features
═══════════════════════════════════════════════════════════════════════════════
"""

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

    def __init__(self, model_id="google/paligemma-3b-mix-224", device="cuda", token=None):
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
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            token=self.token
        )
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            low_cpu_mem_usage=True,
            offload_folder="./offload",
            offload_state_dict=True,
            token=self.token
        ).eval()
        
        # Clear cache after loading
        if self.device == "cuda":
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        logger.info("[VL-JEPA] Model loaded successfully")

    def identify_intent(
        self, frame, prompt="What object is being counted in this video? Answer with a single word.", default_intent="cups"
    ):
        """
        Identify the scanning context using vision-language reasoning.

        Args:
            frame: np.ndarray, PIL.Image or path (H, W, 3)
            prompt: str, the query to ask the director.
            default_intent: str, fallback intent if identification fails.

        Returns:
            str: The identified intent/SKU name.
        """
        if isinstance(frame, torch.Tensor):
            # Convert torch to PIL for processor
            if frame.shape[0] == 3:  # C, H, W -> H, W, C
                frame = frame.permute(1, 2, 0)
            img = Image.fromarray((frame.cpu().numpy() * 255).astype("uint8"))
        elif isinstance(frame, str):
            img = Image.open(frame)
        else:
            img = Image.fromarray(frame)

        inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=20, 
                temperature=0.1,  # Lower temperature for more consistent results
                top_p=0.9,       # Nucleus sampling for better quality
                do_sample=False  # Deterministic generation
            )

        # Output logic
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        
        # Clean the output
        intent = output_text[len(prompt) :].strip().lower()
        
        # Handle cases where output is empty or not useful
        if not intent or len(intent) > 20 or any(char in intent for char in [',', '.', ';', '!', '?']):
            # Extract first word if multiple words are returned
            intent_words = intent.split()
            if intent_words:
                intent = intent_words[0]
            else:
                intent = default_intent
        
        logger.info(f"[VL-JEPA] Identified Intent: {intent}")
        return intent

    def extract_visual_embeddings(self, frame):
        """
        Extract visual latent features for the JEPA world model.

        Returns:
            torch.Tensor: High-dimensional visual features.
        """
        # We can extract the 'vision_tower' hidden states which is equivalent
        # to the X-Encoder in the VL-JEPA paper.
        if not isinstance(frame, (Image.Image, str)):
            img = Image.fromarray(frame)
        else:
            img = frame

        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            vision_outputs = self.model.vision_tower(inputs.pixel_values)
            # PaliGemma vision tower usually outputs (B, L, D)
            features = vision_outputs.last_hidden_state

        return features
