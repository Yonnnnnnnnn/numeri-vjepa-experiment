import sys
import os
import torch

# Force flush stdout immediately
sys.stdout.reconfigure(line_buffering=True)

# Add the implementation directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Implementation'))

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Python version:", sys.version)

# Test the monkey patch directly first
print("\n=== Testing Direct Monkey Patch ===")
print("Creating tensor...")
tensor = torch.tensor([1.0, 2.0, 3.0])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Tensor: {tensor}")
print(f"Device: {device}")

# Import and apply the patch manually
print("\nApplying monkey patch...")
from v2_logic.models.count_gd_engine import CountGDEngine
# Create an instance to trigger patch application
engine = CountGDEngine()
print("Monkey patch applied!")

# Test the problematic pattern
print("\n=== Testing Problematic Pattern ===")
try:
    result = tensor.to(dtype=device)
    print(f"✓ Success! to(dtype=device) works now: {result}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test normal operation
print("\n=== Testing Normal Operation ===")
try:
    result = tensor.to(device)
    print(f"✓ Success! Normal to(device) works: {result}")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test with CountGDEngine
print("\n=== Testing CountGDEngine ===")
try:
    print("Creating dummy image...")
    import numpy as np
    dummy_image = np.ones((480, 640, 3), dtype=np.uint8)
    
    print("Calling count()...")
    count, boxes = engine.count(dummy_image, prompt="items")
    print(f"✓ Success! Count: {count}, Boxes: {boxes}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test Complete ===")
