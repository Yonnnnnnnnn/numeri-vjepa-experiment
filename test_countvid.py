import sys
import os

# Add the Implementation directory to path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "Implementation"))
if base_dir not in sys.path:
    sys.path.append(base_dir)

# Add Techs paths
techs_paths = [
    os.path.join(os.path.dirname(__file__), "Techs/CountVid-main/CountVid-main"),
    os.path.join(os.path.dirname(__file__), "Techs/sam2-main/sam2-main"),
]

for p in techs_paths:
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

from v2_logic.models.count_vid_engine import CountVidEngine

def test_countvid_initialization():
    """Test if CountVidEngine can initialize without errors"""
    print("Testing CountVidEngine initialization...")
    
    try:
        engine = CountVidEngine(device="cpu")
        print("✓ CountVidEngine initialized successfully")
        
        if engine.model is None:
            print("⚠️  CountVidEngine is using mock counting (expected if checkpoints are missing)")
        else:
            print("✓ CountVidEngine loaded real model")
        
        return True
    except Exception as e:
        print(f"✗ Error initializing CountVidEngine: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_countvid_initialization()
    sys.exit(0 if success else 1)
