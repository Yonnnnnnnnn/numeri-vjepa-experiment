"""
BertModelWarper Robustness Unit Test

Verifies that the Smart Dispatcher fallback logic works correctly when
underlying transformers methods are missing.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : TestBertWarperRobustness (this module)

Non-Terminals   :
  ┌─ INTERNAL ────────────────────────────────────────────────────────────────┐
  │  <TestClass>      → class TestBertWarperRobustness(unittest.TestCase)    │
  │  <MockModel>      → class MockBertModel                                  │
  │  <MockConfig>     → class MockBertConfig                                 │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL ────────────────────────────────────────────────────────────────┐
  │  <BertModelWarper> ← from models.GroundingDINO.bertwarper                │
  │  <unittest>       ← from unittest                                         │
  │  <torch>          ← from torch                                            │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : int, str, bool, torch.float32

Production Rules:
  TestBertWarperRobustness → imports + <MockConfig> + <MockModel> + <TestClass>
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import os
import unittest
import torch
from unittest.mock import MagicMock

# Add path to CountVid root so we can import models.GroundingDINO.bertwarper
COUNTVID_ROOT = (
    r"d:\Antigravity\Test VJEPA EVENTBASED LLM\Techs\CountVid-main\CountVid-main"
)
sys.path.append(COUNTVID_ROOT)

# We need to mock transformers if it's not installed or if we want to control it
# But here we want to test our local class, so we import it.
try:
    from models.GroundingDINO.bertwarper import BertModelWarper
except ImportError:
    # Try alternate path if structure is different
    sys.path.append(os.path.join(COUNTVID_ROOT, "models", "GroundingDINO"))
    from bertwarper import BertModelWarper


class MockBertConfig:
    def __init__(self):
        self.output_attentions = False
        self.output_hidden_states = False
        self.use_return_dict = True
        self.is_decoder = False
        self.use_cache = False
        self.num_hidden_layers = 12
        self.hidden_size = 768


class MockBertModel:
    def __init__(self):
        self.config = MockBertConfig()
        self.embeddings = MagicMock()
        self.encoder = MagicMock()
        self.pooler = MagicMock()
        self.dtype = torch.float32

    # Deliberately NOT implementing get_head_mask and invert_attention_mask
    # to trigger the fallback logic in BertModelWarper


class TestBertWarperRobustness(unittest.TestCase):
    def setUp(self):
        self.mock_bert = MockBertModel()
        self.warper = BertModelWarper(self.mock_bert)

    def test_get_head_mask_fallback(self):
        print("\nTesting get_head_mask fallback...")
        # 1. Test with None head_mask
        result = self.warper.get_head_mask(None, 12)
        self.assertEqual(len(result), 12)
        self.assertTrue(all(x is None for x in result))
        print(" -> Passed None input test")

        # 2. Test with Tensor head_mask (1D)
        head_mask = torch.ones(12)
        result = self.warper.get_head_mask(head_mask, 12)
        # Expected shape: [num_hidden_layers, batch, num_heads, seq_len, seq_len]
        # logic: unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) -> [1, 1, 12, 1, 1]
        # expanded: [12, 1, 12, 1, 1]
        self.assertEqual(result.shape, (12, 1, 12, 1, 1))
        print(" -> Passed 1D Tensor input test")

    def test_invert_attention_mask_fallback(self):
        print("\nTesting invert_attention_mask fallback...")
        # Test with 2D mask [batch, seq_len]
        mask = torch.zeros((2, 10))
        result = self.warper.invert_attention_mask(mask)

        # Check shape expansion: [batch, 1, 1, seq_len]
        self.assertEqual(result.shape, (2, 1, 1, 10))

        # Check values: (1.0 - 0.0) * min_float -> large negative
        self.assertTrue(result[0, 0, 0, 0] < -1000)
        print(" -> Passed 2D mask test")

        # Test with 3D mask [batch, seq_len, seq_len]
        mask3d = torch.zeros((2, 10, 10))
        result3d = self.warper.invert_attention_mask(mask3d)
        self.assertEqual(result3d.shape, (2, 1, 10, 10))
        print(" -> Passed 3D mask test")


if __name__ == "__main__":
    unittest.main()
