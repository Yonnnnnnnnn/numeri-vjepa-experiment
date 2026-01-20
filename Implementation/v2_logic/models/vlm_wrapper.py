"""
VLM Wrapper Module

Wrapper for Qwen2.5-VL Vision-Language Model for zero-shot inventory counting.
Uses 4-bit quantization for T4 GPU compatibility.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : VLMWrapper (this file)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <VLMInferenceModel>  → Main wrapper class for VLM inference              │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <Qwen2VLForConditionalGeneration>  ← from transformers (VLM backbone)    │
  │  <AutoProcessor>                    ← from transformers (image processing)│
  │  <BitsAndBytesConfig>               ← from transformers (4-bit quant)     │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : torch, str, int, float

Production Rules:
  VLMWrapper      → imports + <VLMInferenceModel>
  <VLMInferenceModel> → __init__ + predict
═══════════════════════════════════════════════════════════════════════════════

Pattern: Facade
- Simplifies complex VLM loading and inference into a single interface
- Hides quantization, preprocessing, and prompt construction details
"""

import torch
from PIL import Image
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    BitsAndBytesConfig,
)


class VLMInferenceModel:
    """
    Vision-Language Model wrapper for inventory counting.
    Uses Qwen2.5-VL with 4-bit quantization for T4 GPU compatibility.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
    ):
        print("Loading VLM (Qwen2.5-VL) components...")

        # 4-bit Quantization Config for T4 GPU
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        print(f" - Loading Model: {model_id} (4-bit)")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        print(" - Loading Processor...")
        self.processor = Qwen2VLProcessor.from_pretrained(model_id)

        self.device = self.model.device
        print(f"VLM loaded on: {self.device}")

    def predict(
        self,
        rgb_frame,
        prompt_text: str = "Count all the visible inventory items in this image. Provide a number.",
    ) -> str:
        """
        Run VLM inference on an RGB frame.

        Args:
            rgb_frame: numpy array (H, W, C) in RGB format, values 0-255
            prompt_text: The counting prompt

        Returns:
            Generated text response from the VLM
        """
        # Convert numpy to PIL Image
        if isinstance(rgb_frame, torch.Tensor):
            rgb_frame = rgb_frame.squeeze().permute(1, 2, 0).cpu().numpy()
            rgb_frame = (rgb_frame * 255).astype("uint8")

        image = Image.fromarray(rgb_frame)

        # Construct chat-style message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # Prepare inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
            )

        # Decode (remove input tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text


if __name__ == "__main__":
    # Quick test
    model = VLMInferenceModel()
    print("Model loaded successfully.")
