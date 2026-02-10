"""
Environment verification script for CUDA, MPS, CPU, CLIP, and ultralytics
Supports multiple devices: NVIDIA CUDA, Apple MPS, and CPU
"""

import sys

print("=" * 60)
print("Environment Verification Test")
print("=" * 60)

# Device detection function
def get_device():
    """Detect and return the best available device"""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps"), "MPS"
    else:
        return torch.device("cpu"), "CPU"

# Test 1: Device availability
print("\n[1] Testing device availability...")
try:
    import torch
    print(f"  ✓ PyTorch version: {torch.__version__}")
    
    device, device_name = get_device()
    print(f"  ✓ Selected device: {device_name} ({device})")
    
    if device_name == "CUDA":
        print(f"  ✓ CUDA version: {torch.version.cuda}")
        print(f"  ✓ Number of GPUs: {torch.cuda.device_count()}")
        print(f"  ✓ Current GPU: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    elif device_name == "MPS":
        print(f"  ✓ Apple Metal Performance Shaders (MPS) is available")
        print(f"  ✓ Using Apple Silicon GPU acceleration")
    else:
        print(f"  ✓ Using CPU (no GPU acceleration available)")
        if torch.cuda.is_available():
            print(f"  ⚠ Note: CUDA is available but not selected")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"  ⚠ Note: MPS is available but not selected")
    
    # Test tensor creation on selected device
    test_tensor = torch.randn(2, 3).to(device)
    print(f"  ✓ Successfully created tensor on {device_name}: {test_tensor.shape}")
    print(f"  ✓ Tensor device: {test_tensor.device}")
    
except Exception as e:
    print(f"  ✗ Error testing device: {e}")
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

# Test 4: Integration test - create tensors on selected device
print("\n[4] Testing tensor operations on selected device...")
try:
    # Get device
    device, device_name = get_device()
    
    # Create tensors on selected device
    a = torch.randn(3, 3).to(device)
    b = torch.randn(3, 3).to(device)
    c = torch.matmul(a, b)
    print(f"  ✓ Successfully performed matrix multiplication on {device_name}")
    print(f"  ✓ Result shape: {c.shape}")
    print(f"  ✓ Result device: {c.device}")
    
except Exception as e:
    print(f"  ✗ Error testing tensor operations: {e}")
    sys.exit(1)

# Final summary
print("\n" + "=" * 60)
print("Verification Summary")
print("=" * 60)
device, device_name = get_device()
print(f"✓ Device: {device_name} ({device}) - Available and working")
print("✓ CLIP: Imported and functional")
print("✓ ultralytics: Imported and functional")
print("✓ All components are ready for KDEOV project")
print("=" * 60)
print("\nEnvironment verification completed successfully!")
print(f"Using device: {device_name} for training and inference.")
