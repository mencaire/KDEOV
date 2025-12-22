"""
Temporary script to verify CUDA, CLIP, and ultralytics availability
This script will be deleted after verification
"""

import sys

print("=" * 60)
print("Environment Verification Test")
print("=" * 60)

# Test 1: CUDA availability
print("\n[1] Testing CUDA availability...")
try:
    import torch
    print(f"  ✓ PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"  ✓ CUDA is available")
        print(f"  ✓ CUDA version: {torch.version.cuda}")
        print(f"  ✓ Number of GPUs: {torch.cuda.device_count()}")
        print(f"  ✓ Current GPU: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Test tensor creation on CUDA
        test_tensor = torch.randn(2, 3).cuda()
        print(f"  ✓ Successfully created tensor on CUDA: {test_tensor.shape}")
    else:
        print("  ✗ CUDA is NOT available")
        print("  Error: This script requires CUDA support")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ Error testing CUDA: {e}")
    sys.exit(1)

# Test 2: CLIP import and basic functionality
print("\n[2] Testing CLIP...")
try:
    import clip
    print(f"  ✓ CLIP module imported successfully")
    print(f"  ✓ CLIP location: {clip.__file__}")
    
    # Test available models (doesn't download)
    available_models = clip.available_models()
    print(f"  ✓ Available CLIP models: {len(available_models)}")
    print(f"  ✓ Sample models: {', '.join(available_models[:3])}...")
    
    # Test tokenizer
    test_text = "a photo of a cat"
    tokens = clip.tokenize([test_text])
    print(f"  ✓ Tokenizer works: tokenized '{test_text}' -> shape {tokens.shape}")
    
except ImportError as e:
    print(f"  ✗ Failed to import CLIP: {e}")
    print("  Please install: pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)
except Exception as e:
    print(f"  ✗ Error testing CLIP: {e}")
    sys.exit(1)

# Test 3: Ultralytics import
print("\n[3] Testing ultralytics (YOLOv8)...")
try:
    import ultralytics
    print(f"  ✓ ultralytics module imported successfully")
    print(f"  ✓ ultralytics location: {ultralytics.__file__}")
    
    if hasattr(ultralytics, '__version__'):
        print(f"  ✓ ultralytics version: {ultralytics.__version__}")
    
    # Test YOLO class import
    from ultralytics import YOLO
    print(f"  ✓ YOLO class imported successfully")
    print(f"  ✓ YOLO class is available for use")
    
except ImportError as e:
    print(f"  ✗ Failed to import ultralytics: {e}")
    print("  Please install: pip install ultralytics")
    sys.exit(1)
except Exception as e:
    print(f"  ✗ Error testing ultralytics: {e}")
    sys.exit(1)

# Test 4: Integration test - create a simple tensor on CUDA
print("\n[4] Testing CUDA tensor operations...")
try:
    # Create tensors on CUDA
    a = torch.randn(3, 3).cuda()
    b = torch.randn(3, 3).cuda()
    c = torch.matmul(a, b)
    print(f"  ✓ Successfully performed matrix multiplication on CUDA")
    print(f"  ✓ Result shape: {c.shape}")
    
except Exception as e:
    print(f"  ✗ Error testing CUDA operations: {e}")
    sys.exit(1)

# Final summary
print("\n" + "=" * 60)
print("Verification Summary")
print("=" * 60)
print("✓ CUDA: Available and working")
print("✓ CLIP: Imported and functional")
print("✓ ultralytics: Imported and functional")
print("✓ All components are ready for KDEOV project")
print("=" * 60)
print("\nEnvironment verification completed successfully!")
print("You can proceed with model training and inference.")
