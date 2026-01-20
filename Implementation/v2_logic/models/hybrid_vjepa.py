"""
Hybrid V-JEPA Model Definition
Combines Frozen V-JEPA 2 Backbone (Event) + RGB Encoder (CLIP/SigLIP) + Zero-Shot LLM Head.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPVisionModel,
    CLIPImageProcessor,
    BitsAndBytesConfig,
)


class HybridInferenceModel(nn.Module):
    def __init__(
        self,
        llm_model_id="lmsys/vicuna-7b-v1.5",
        vision_backbone_id="openai/clip-vit-large-patch14",
    ):
        super().__init__()
        print(f"Loading Hybrid Model components...")

        # 1. Visual Backbone for RGB (Static Labels)
        # Using CLIP as a proxy for "RGB Encoder" in this prototype
        print(f" - Loading RGB Encoder: {vision_backbone_id}")
        self.rgb_encoder = CLIPVisionModel.from_pretrained(vision_backbone_id)
        self.rgb_processor = CLIPImageProcessor.from_pretrained(vision_backbone_id)

        # 2. V-JEPA 2 Backbone for Event (Motion/Dynamics)
        print(f" - Loading Event Encoder (Real V-JEPA Architecture)...")
        try:
            # Import from local source (vjepa_src)
            from vjepa_src.models.vision_transformer import vit_huge

            # Instantiate V-JEPA Huge (Random Weights)
            # Default to 1 frame for this event grid prototype,
            # In production with real checkpoints, match the checkpoint config (e.g. 16 frames)
            self.event_encoder = vit_huge(img_size=224, patch_size=14, num_frames=1)
            self.embed_dim_event = 1280  # ViT-Huge dim
            print("   -> Success: Loaded V-JEPA ViT-Huge from local source.")
        except ImportError as e:
            print(f"   -> Failed to load local V-JEPA: {e}")
            print("   -> Fallback to CLIP Proxy.")
            self.event_encoder = CLIPVisionModel.from_pretrained(vision_backbone_id)
            self.embed_dim_event = self.event_encoder.config.hidden_size

        # 3. Projectors (Mapping Visual Dim -> LLM Dim)
        # Assuming LLM dim is 4096 (Llama 7B standard)
        self.visual_hidden_size_rgb = self.rgb_encoder.config.hidden_size
        self.llm_hidden_size = 4096

        self.event_projector = nn.Linear(self.embed_dim_event, self.llm_hidden_size)
        self.rgb_projector = nn.Linear(
            self.visual_hidden_size_rgb, self.llm_hidden_size
        )

        # 4. LLM Head (Zero-Shot)
        print(f" - Loading LLM Head: {llm_model_id}")
        # Using 4-bit loading if possible for memory efficiency on test environment
        # For L40S, we can load full precision or bf16
        try:
            # OPTIMIZATION FOR COLAB/T4: Use 4-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print("   -> Attempting 4-bit Load (bitsandbytes)...")
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_id, quantization_config=bnb_config, device_map="auto"
            )
        except Exception as e:
            print(
                f"Warning: Could not load LLM with 4-bit/device_map='auto'. Fallback to CPU/Default. Error: {e}"
            )
            # Fallback to standard loading if bitsandbytes missing
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_id,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
            )

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_id)

        # DTYPE UNIFICATION: Convert vision components to float16 for consistency
        self.rgb_encoder = self.rgb_encoder.half()
        self.event_encoder = self.event_encoder.half()
        self.event_projector = self.event_projector.half()
        self.rgb_projector = self.rgb_projector.half()

        # Freeze Backbones (Zero-Shot Strategy)
        self.freeze_backbones()

    def freeze_backbones(self):
        """Freeze all weights except projectors (if we were training, but for Zero-Shot everything is frozen)"""
        for param in self.rgb_encoder.parameters():
            param.requires_grad = False
        for param in self.event_encoder.parameters():
            param.requires_grad = False
        for param in self.llm.parameters():
            param.requires_grad = False

        # For Zero-Shot, we typically use random initialized projectors or pre-trained ones.
        # Since we don't have pre-trained projectors for this specific setup,
        # the inference results will be garbage (random noise) unless we load specific adapter weights.
        # This is expected in the "Architecture Prototype" stage.

    def forward_features(self, event_grid, rgb_keyframe):
        """
        Extract features from both towers.
        Args:
            event_grid: (B, C, H, W)
            rgb_keyframe: (B, C, H, W)
        """
        # Resize inputs to 224x224 for CLIP Compatibility
        # Using bilinear interpolation
        target_size = (224, 224)
        event_resized = nn.functional.interpolate(
            event_grid, size=target_size, mode="bilinear", align_corners=False
        )
        rgb_resized = nn.functional.interpolate(
            rgb_keyframe, size=target_size, mode="bilinear", align_corners=False
        )

        # 1. Event Stream -> V-JEPA
        if hasattr(self, "embed_dim_event") and self.embed_dim_event == 1280:
            # V-JEPA Native forward (converted to float16 already)
            event_emb = self.event_encoder(event_resized)  # (B, N, D)
        else:
            # CLIP Proxy
            event_outputs = self.event_encoder(pixel_values=event_resized)
            event_emb = event_outputs.last_hidden_state  # (B, Seq, Dim)

        # 2. RGB Stream -> CLIP
        rgb_outputs = self.rgb_encoder(pixel_values=rgb_resized)
        rgb_emb = rgb_outputs.last_hidden_state  # (B, Seq, Dim)

        return event_emb, rgb_emb

    def predict(self, event_grid, rgb_keyframe, prompt_text="Count inventory items."):
        """
        End-to-End Inference Generation
        """
        device = self.llm.device

        # Ensure inputs are on correct device
        event_grid = (
            event_grid.to(device).half()
            if self.llm.dtype == torch.float16
            else event_grid.to(device)
        )
        rgb_keyframe = (
            rgb_keyframe.to(device).half()
            if self.llm.dtype == torch.float16
            else rgb_keyframe.to(device)
        )

        with torch.no_grad():
            # 1. Extract & Project
            event_emb, rgb_emb = self.forward_features(event_grid, rgb_keyframe)

            event_tokens = self.event_projector(event_emb)  # (B, Seq, 4096)
            rgb_tokens = self.rgb_projector(rgb_emb)  # (B, Seq, 4096)

            # 2. Fuse (Concatenate for simplicity in Zero-Shot)
            # Structure: [RGB_TOKENS] + [EVENT_TOKENS]
            visual_inputs = torch.cat([rgb_tokens, event_tokens], dim=1)

            # 3. Construct Text Embeddings
            text_inputs = self.tokenizer(prompt_text, return_tensors="pt").to(device)
            input_ids = text_inputs.input_ids
            text_emb = self.llm.get_input_embeddings()(input_ids)

            # 4. Combine Visual + Text
            # Ensure visual tokens match LLM embedding dtype
            visual_inputs = visual_inputs.to(text_emb.dtype)
            # [VISUAL] + [TEXT]
            final_inputs_embeds = torch.cat([visual_inputs, text_emb], dim=1)

            # 5. Generate
            outputs = self.llm.generate(
                inputs_embeds=final_inputs_embeds,
                max_new_tokens=128,
                temperature=0.1,  # Low temp for factual counting
            )

            # Decode output (skipping input tokens is tricky with inputs_embeds,
            # usually we just decode the whole thing and slice)
            decoded_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            return decoded_text
