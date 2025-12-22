"""
Simple script to list all available CLIP models
Only lists model names without downloading them
"""

def list_clip_models():
    """
    List all available CLIP models (without downloading)
    """
    try:
        import clip
    except ImportError:
        print("Error: CLIP module not found.")
        print("Please install CLIP using:")
        print("  pip install git+https://github.com/openai/CLIP.git")
        return
    
    print("=" * 60)
    print("Available CLIP Models")
    print("=" * 60)
    
    # Get list of available models (this doesn't download anything)
    available_models = clip.available_models()
    
    print(f"\nTotal number of available models: {len(available_models)}\n")
    
    # Print each model
    print("All Available Models:")
    for i, model_name in enumerate(available_models, 1):
        print(f"  {i}. {model_name}")
    
    print("\n" + "=" * 60)
    print("Model Categories")
    print("=" * 60)
    
    # Categorize models
    resnet_models = [m for m in available_models if m.startswith('RN')]
    vit_models = [m for m in available_models if m.startswith('ViT')]
    
    if resnet_models:
        print("\nResNet-based Models:")
        for model in resnet_models:
            print(f"  - {model}")
    
    if vit_models:
        print("\nVision Transformer (ViT) Models:")
        for model in vit_models:
            print(f"  - {model}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    try:
        list_clip_models()
    except ImportError:
        print("Error: CLIP module not found.")
        print("Please install CLIP using:")
        print("  pip install git+https://github.com/openai/CLIP.git")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
