import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch

# Import the class we just modified
import sys
import os

# Add the project root to sys.path if needed
sys.path.append(os.path.join(os.getcwd(), "Implementation"))

try:
    from v2_logic.models.vlm_wrapper import VLMInferenceModel
except ImportError:
    # Fallback for different path structures
    sys.path.append(os.getcwd())
    from Implementation.v2_logic.models.vlm_wrapper import VLMInferenceModel


class TestVLMWrapperInterface(unittest.TestCase):
    @patch("transformers.Qwen2VLForConditionalGeneration.from_pretrained")
    @patch("transformers.Qwen2VLProcessor.from_pretrained")
    def test_predict_with_additional_args(self, mock_processor_load, mock_model_load):
        # Setup mocks
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model_load.return_value = mock_model

        mock_processor = MagicMock()
        mock_processor_load.return_value = mock_processor

        # Initialize model (mocked)
        vlm = VLMInferenceModel()

        # Mock processor behavior
        mock_processor.apply_chat_template.return_value = "templated_text"
        mock_processor.return_value = MagicMock()
        mock_processor.return_value.to.return_value = MagicMock(input_ids=[[1, 2, 3]])

        # Mock model generate
        mock_model.generate.return_value = [[1, 2, 3, 4, 5]]

        # Mock decode
        mock_processor.batch_decode.return_value = ["result text"]

        # Test frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Call predict with max_new_tokens (this is what caused the bug)
        result = vlm.predict(frame, prompt_text="test prompt", max_new_tokens=50)

        # Check if generate was called with max_new_tokens=50
        args, kwargs = mock_model.generate.call_args
        self.assertEqual(kwargs.get("max_new_tokens"), 50)
        self.assertEqual(result, "result text")
        print("✅ Interface verification PASSED: max_new_tokens passed to generate()")

    @patch("transformers.Qwen2VLForConditionalGeneration.from_pretrained")
    @patch("transformers.Qwen2VLProcessor.from_pretrained")
    def test_predict_default_args(self, mock_processor_load, mock_model_load):
        # Setup mocks
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model_load.return_value = mock_model

        mock_processor = MagicMock()
        mock_processor_load.return_value = mock_processor

        vlm = VLMInferenceModel()

        mock_processor.apply_chat_template.return_value = "templated_text"
        mock_processor.return_value = MagicMock()
        mock_processor.return_value.to.return_value = MagicMock(input_ids=[[1, 2, 3]])
        mock_model.generate.return_value = [[1, 2, 3, 4, 5]]
        mock_processor.batch_decode.return_value = ["default result"]

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Call predict without extra args
        result = vlm.predict(frame, prompt_text="test prompt")

        # Check if default max_new_tokens=128 was used
        args, kwargs = mock_model.generate.call_args
        self.assertEqual(kwargs.get("max_new_tokens"), 128)
        self.assertEqual(result, "default result")
        print("✅ Interface verification PASSED: default max_new_tokens=128 used")


if __name__ == "__main__":
    unittest.main()
