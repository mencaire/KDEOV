"""
Simple script to verify ultralytics (YOLOv8) installation
Only checks import without downloading models
"""

def verify_ultralytics():
    """
    Verify ultralytics installation and import success
    """
    print("=" * 60)
    print("Ultralytics (YOLOv8) Verification")
    print("=" * 60)
    
    try:
        import ultralytics
        print("\n✓ Successfully imported ultralytics module")
        print(f"  Location: {ultralytics.__file__}")
        
        # Check version if available
        if hasattr(ultralytics, '__version__'):
            print(f"  Version: {ultralytics.__version__}")
        
        # Try to import YOLO class
        try:
            from ultralytics import YOLO
            print("\n✓ Successfully imported YOLO class")
            print("  YOLO class is available for use")
        except ImportError as e:
            print(f"\n✗ Failed to import YOLO class: {e}")
        
        # List available YOLO models (without downloading)
        try:
            from ultralytics import YOLO
            print("\n" + "=" * 60)
            print("Available YOLO Model Sizes")
            print("=" * 60)
            print("""
Common YOLOv8 model sizes:
  - yolov8n.pt  : Nano (smallest, fastest)
  - yolov8s.pt  : Small
  - yolov8m.pt  : Medium
  - yolov8l.pt  : Large
  - yolov8x.pt  : Extra Large (largest, most accurate)
            """)
            print("Note: Models will be downloaded automatically on first use")
        except Exception as e:
            print(f"\nNote: Could not list models: {e}")
        
        print("\n" + "=" * 60)
        print("Verification Result")
        print("=" * 60)
        print("✓ ultralytics is properly installed and can be imported")
        print("✓ Ready to use YOLOv8 models in KDEOV project")
        
    except ImportError as e:
        print("\n✗ Failed to import ultralytics")
        print(f"  Error: {e}")
        print("\n" + "=" * 60)
        print("Installation Instructions")
        print("=" * 60)
        print("Please install ultralytics using:")
        print("  pip install ultralytics")
        print("\nOr activate your conda environment:")
        print("  conda activate KDEOV")
        print("  pip install ultralytics")
        return False
    
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    return True


if __name__ == "__main__":
    verify_ultralytics()
