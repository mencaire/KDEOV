"""
Model Summary and Visualization Script for KDEOV

This script provides comprehensive information about the KDEOV model:
- Model architecture overview
- Parameter counts (trainable vs frozen)
- Component details
- Input/output shapes
- Memory usage estimates

Usage:
    python model_summary.py [--backbone BACKBONE] [--fusion FUSION] [--static]
    
Options:
    --backbone BACKBONE    Backbone type: yolov8n, yolov5s, or simple (default: yolov8n)
    --fusion FUSION        Fusion type: film or cross_attention (default: film)
    --static               Show static summary only (no model loading)
"""

import torch
import torch.nn as nn
import ssl  
ssl._create_default_https_context = ssl._create_unverified_context
from typing import Dict, List, Tuple, Optional
import sys
import os
import argparse

# Add project root to path
sys.path.append(os.getcwd())

# Try to import required modules
MODELS_AVAILABLE = False
KDEOVModel = None
clip = None

try:
    from models import KDEOVModel
    import clip
    MODELS_AVAILABLE = True
except ImportError as e:
    # Silently handle import error - will be handled in main()
    pass


def count_parameters(model: nn.Module, trainable_only: bool = False) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, only count trainable parameters
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if trainable_only:
        return trainable_params, trainable_params
    return total_params, trainable_params


def format_number(num: int) -> str:
    """Format large numbers with K/M/B suffixes."""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)


def get_model_size_mb(model: nn.Module) -> float:
    """Estimate model size in MB (assuming float32)."""
    total_params = sum(p.numel() for p in model.parameters())
    # 4 bytes per float32 parameter
    size_mb = (total_params * 4) / (1024 * 1024)
    return size_mb


def print_section(title: str, width: int = 80):
    """Print a formatted section header."""
    try:
        print("\n" + "=" * width)
        print(f" {title}".center(width))
        print("=" * width)
    except UnicodeEncodeError:
        # Fallback for Windows console encoding issues
        print("\n" + "=" * width)
        print(f" {title}".center(width))
        print("=" * width)


def print_component_summary(
    name: str,
    component: nn.Module,
    is_frozen: bool = False,
    indent: int = 0
):
    """Print summary for a model component."""
    prefix = "  " * indent
    total_params, trainable_params = count_parameters(component)
    
    status = "FROZEN" if is_frozen else "TRAINABLE"
    status_symbol = "[FROZEN]" if is_frozen else "[TRAINABLE]"
    
    print(f"{prefix}{status_symbol} {name} ({status})")
    print(f"{prefix}  Total Parameters: {format_number(total_params)} ({total_params:,})")
    if not is_frozen:
        print(f"{prefix}  Trainable Parameters: {format_number(trainable_params)} ({trainable_params:,})")
    else:
        print(f"{prefix}  Trainable Parameters: 0 (frozen)")
    print(f"{prefix}  Model Size: {get_model_size_mb(component):.2f} MB")
    print()


def analyze_model_components(model: KDEOVModel) -> Dict:
    """Analyze each component of the KDEOV model."""
    components_info = {}
    
    # Text Encoder (Frozen CLIP)
    components_info['text_encoder'] = {
        'name': 'FrozenCLIPTextEncoder',
        'module': model.text_encoder,
        'frozen': True,
        'description': 'Frozen CLIP text encoder for semantic reference'
    }
    
    # Visual Backbone
    components_info['visual_backbone'] = {
        'name': 'LightweightVisualBackbone',
        'module': model.visual_backbone,
        'frozen': False,
        'description': f'YOLO-based visual backbone ({model.visual_backbone.backbone_type})'
    }
    
    # Projection Network
    components_info['projection'] = {
        'name': 'ProjectionNetwork',
        'module': model.projection,
        'frozen': False,
        'description': 'MLP projection to CLIP embedding space'
    }
    
    # Fusion Module
    components_info['fusion_module'] = {
        'name': 'CrossModalFusionModule',
        'module': model.fusion_module,
        'frozen': False,
        'description': f'Cross-modal fusion ({model.fusion_module.fusion_type})'
    }
    
    # Spatial Projection (for open-vocabulary detection)
    components_info['spatial_projection'] = {
        'name': 'SpatialProjection',
        'module': model.spatial_projection,
        'frozen': False,
        'description': 'Per-location projection to embedding space (no global pooling) for detection'
    }
    
    return components_info


def test_forward_pass(model: KDEOVModel, device: torch.device) -> Dict:
    """Test forward pass and capture input/output shapes."""
    if clip is None:
        raise ImportError("CLIP module not available")
    
    model.eval()
    model = model.to(device)
    
    # Test inputs - ensure consistent dtype
    batch_size = 2
    image_size = 224
    
    # Get the dtype from model parameters (usually float32)
    model_dtype = next(model.parameters()).dtype
    
    images = torch.randn(batch_size, 3, image_size, image_size, dtype=model_dtype).to(device)
    text = clip.tokenize(["a photo of a cat", "a photo of a dog"]).to(device)
    
    shapes = {}
    
    with torch.no_grad():
        # Test image encoding
        image_emb = model.encode_image(images)
        shapes['image_input'] = images.shape
        shapes['image_embedding'] = image_emb.shape
        
        # Test text encoding (dtype conversion is handled in FrozenCLIPTextEncoder)
        text_emb = model.encode_text(text)
        shapes['text_input'] = text.shape
        shapes['text_embedding'] = text_emb.shape
        
        # Test forward pass - ensure consistent dtypes
        # Convert text to match model dtype before forward pass
        outputs = model(images=images, text=text, use_fusion=True)
        shapes['forward_outputs'] = {k: v.shape for k, v in outputs.items()}
        
        # Test zero-shot classification
        class_names = ["cat", "dog", "bird"]
        logits = model.zero_shot_classify(images[:1], class_names)
        shapes['zero_shot_logits'] = logits.shape
        
        # Test spatial embeddings (detection path)
        spatial_emb = model.get_spatial_embeddings(images, use_fusion=False)
        shapes['spatial_embeddings'] = spatial_emb.shape
        
        # Test open-vocabulary detection (optional; may fail if e.g. torchvision.ops missing)
        try:
            detections = model.open_vocabulary_detect(
                images[:1], class_names=class_names,
                score_threshold=0.1, max_detections_per_image=5
            )
            shapes['detection_boxes'] = detections[0]['boxes'].shape
            shapes['detection_scores'] = detections[0]['scores'].shape
            shapes['detection_labels_count'] = len(detections[0]['labels'])
        except Exception:
            pass  # detection test optional
    
    return shapes


def print_architecture_diagram(model: KDEOVModel):
    """Print a text-based architecture diagram."""
    print_section("Model Architecture Diagram", width=80)
    
    diagram = f"""
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     KDEOV Model Architecture                             │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Text Stream (Frozen):
    ┌─────────────────┐
    │  Text Input     │  [batch, seq_len]
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Frozen CLIP     │  FROZEN
    │ Text Encoder    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Text Embeddings │  [batch, {model.embedding_dim}]
    └─────────────────┘
    
    Visual Stream (Trainable) - shared backbone and fusion, then two paths:
    ┌─────────────────┐
    │  Image Input    │  [batch, 3, H, W]
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Visual Backbone  │  TRAINABLE ({model.visual_backbone.backbone_type})
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Multi-scale     │  [batch, C, H', W']
    │ Features        │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Fusion Module   │  TRAINABLE ({model.fusion_module.fusion_type}, optional)
    └────────┬────────┘
             │
       ┌─────┴─────┐
       │           │
       ▼           ▼
    ┌──────────────────┐   ┌──────────────────┐
    │ Projection       │   │ Spatial           │   TRAINABLE
    │ Network          │   │ Projection        │
    │ (global pool)    │   │ (no pool)         │
    └────────┬─────────┘   └────────┬──────────┘
             │                      │
             ▼                      ▼
    ┌──────────────────┐   ┌──────────────────┐
    │ Image Embeddings  │   │ Spatial Embeddings│  [batch, {model.embedding_dim}, Hf, Wf]
    │ [batch, {model.embedding_dim}] │   │ (per-location)     │
    └────────┬─────────┘   └────────┬──────────┘
             │                      │
             ▼                      ▼
    ┌──────────────────┐   ┌──────────────────┐
    │ Classification / │   │ Open-vocabulary  │
    │ retrieval        │   │ detection        │  boxes, scores, labels
    │ (similarity)     │   │ (grid + NMS)      │
    └──────────────────┘   └──────────────────┘
    """
    print(diagram)


def print_detailed_summary(model: KDEOVModel, device: torch.device):
    """Print detailed model summary."""
    print_section("KDEOV Model Summary", width=80)
    
    # Overall statistics
    total_params, trainable_params = count_parameters(model)
    frozen_params = total_params - trainable_params
    
    print(f"\n[STATS] Overall Statistics:")
    print(f"  Total Parameters: {format_number(total_params)} ({total_params:,})")
    print(f"  Trainable Parameters: {format_number(trainable_params)} ({trainable_params:,})")
    print(f"  Frozen Parameters: {format_number(frozen_params)} ({frozen_params:,})")
    print(f"  Model Size: {get_model_size_mb(model):.2f} MB")
    print(f"  Device: {device}")
    print(f"  Embedding Dimension: {model.embedding_dim}")
    
    # Component details
    print_section("Component Details", width=80)
    components_info = analyze_model_components(model)
    
    for key, info in components_info.items():
        print_component_summary(
            info['name'],
            info['module'],
            is_frozen=info['frozen'],
            indent=0
        )
        if info['description']:
            print(f"  Description: {info['description']}")
        print()
    
    # Input/Output shapes
    print_section("Input/Output Shapes", width=80)
    try:
        shapes = test_forward_pass(model, device)
        
        print("\n[INPUT] Input Shapes:")
        print(f"  Images: {shapes['image_input']}")
        print(f"  Text tokens: {shapes['text_input']}")
        
        print("\n[OUTPUT] Output Shapes:")
        print(f"  Image embeddings: {shapes['image_embedding']}")
        print(f"  Text embeddings: {shapes['text_embedding']}")
        
        print("\n[FORWARD] Forward Pass Outputs:")
        for key, shape in shapes['forward_outputs'].items():
            print(f"  {key}: {shape}")
        
        print(f"\n[ZERO-SHOT] Zero-shot Classification:")
        print(f"  Logits shape: {shapes['zero_shot_logits']}")
        
        if 'spatial_embeddings' in shapes:
            print(f"\n[DETECTION] Open-vocabulary detection path:")
            print(f"  Spatial embeddings: {shapes['spatial_embeddings']}")
            if 'detection_boxes' in shapes:
                print(f"  Detection boxes (image 0): {shapes['detection_boxes']}")
                print(f"  Detection scores (image 0): {shapes['detection_scores']}")
                print(f"  Detection labels count (image 0): {shapes['detection_labels_count']}")
        
    except Exception as e:
        print(f"[WARNING] Could not test forward pass: {e}")
        print("   This is normal if CUDA/CLIP is not available")
    
    # Training information
    print_section("Training Information", width=80)
    print("\n[TRAINING] Training Components:")
    print("  [OK] Visual Backbone: Trainable")
    print("  [OK] Projection Network: Trainable (classification/retrieval path)")
    print("  [OK] Spatial Projection: Trainable (detection path)")
    print("  [OK] Fusion Module: Trainable")
    print("  [FROZEN] Text Encoder: Frozen (CLIP)")
    print("\n[LOSS] Loss Functions:")
    print("  - DistillationLoss: Aligns student with teacher CLIP")
    print("  - CrossModalAlignmentLoss: Contrastive image-text alignment")
    print("  - FeatureAlignmentLoss: Combined loss for training")


def print_model_comparison():
    """Print comparison with typical CLIP model sizes."""
    print_section("Model Size Comparison", width=80)
    
    comparison = """
    Model Size Reference:
    
    CLIP ViT-B/32 (Full Model):
      - Parameters: ~150M
      - Size: ~600 MB
    
    KDEOV Model (Student):
      - Parameters: ~5-10M (estimated, depends on backbone)
      - Size: ~20-40 MB (estimated)
      - Reduction: ~95% smaller than full CLIP
    
    Benefits:
      ✓ Much smaller model size
      ✓ Faster inference
      ✓ Lower memory footprint
      ✓ Maintains semantic alignment with CLIP
    """
    print(comparison)


def print_static_summary():
    """Print static architecture summary without loading the model."""
    print_section("KDEOV Model Architecture (Static Summary)", width=80)
    
    architecture = """
    Model Components:
    
    1. FrozenCLIPTextEncoder [FROZEN]
       - Purpose: Semantic reference from pretrained CLIP
       - Input: Tokenized text [batch, seq_len]
       - Output: Text embeddings [batch, 512]
       - Parameters: ~63M (frozen, from CLIP ViT-B/32)
    
    2. LightweightVisualBackbone [TRAINABLE]
       - Purpose: Efficient visual feature extraction
       - Options: YOLOv8n, YOLOv5s, or simple CNN fallback
       - Input: Images [batch, 3, H, W]
       - Output: Multi-scale features [batch, C, H', W']
       - Parameters: ~2-5M (depends on backbone)
    
    3. ProjectionNetwork [TRAINABLE]
       - Purpose: Map visual features to CLIP embedding space (classification/retrieval path)
       - Architecture: 2-layer MLP with LayerNorm; global pooling before projection
       - Input: Visual features [batch, C, H', W'] -> pooled to [batch, C]
       - Output: Image embeddings [batch, 512]
       - Parameters: ~0.5-1M
    
    4. CrossModalFusionModule [TRAINABLE]
       - Purpose: Fuse text and image features
       - Types: FiLM or Cross-Attention
       - Input: Image features + Text embeddings
       - Output: Fused features
       - Parameters: ~0.5-1M
    
    5. SpatialProjection [TRAINABLE]
       - Purpose: Per-location projection for open-vocabulary detection (no global pooling)
       - Architecture: 1x1 conv + GroupNorm; preserves spatial dims
       - Input: Visual features [batch, C, H', W']
       - Output: Spatial embeddings [batch, 512, H', W']
       - Parameters: ~0.3-0.5M
    
    Two visual output paths (shared backbone + fusion):
      - Classification/retrieval: backbone -> fusion -> ProjectionNetwork -> image embeddings
      - Detection: backbone -> fusion -> SpatialProjection -> spatial embeddings -> grid + NMS
    
    Total Estimated Parameters:
      - Trainable: ~4-8M
      - Frozen (CLIP): ~63M
      - Total: ~67-71M
    
    Model Size: ~25-35 MB (trainable only)
    """
    print(architecture)
    
    print_section("Training Loss Functions", width=80)
    losses = """
    1. DistillationLoss
       - Aligns student image embeddings with teacher CLIP embeddings
       - Types: Cosine similarity or L2 loss
    
    2. CrossModalAlignmentLoss
       - Contrastive loss (InfoNCE) for image-text alignment
       - Ensures matched pairs have high similarity
    
    3. FeatureAlignmentLoss
       - Combined loss: distillation + alignment
       - Configurable weights for each component
    """
    print(losses)
    
    print_section("Usage", width=80)
    usage = """
    Basic Usage:
    
    from models import KDEOVModel
    import torch
    import clip
    
    # Initialize
    model = KDEOVModel(
        clip_model_name="ViT-B/32",
        backbone_type="yolov8n",
        fusion_type="film"
    )
    
    # Encode
    images = torch.randn(4, 3, 224, 224)
    text = clip.tokenize(["a photo of a cat"])
    
    image_emb = model.encode_image(images)
    text_emb = model.encode_text(text)
    
    # Zero-shot classification
    logits = model.zero_shot_classify(images, ["cat", "dog", "bird"])
    
    # Open-vocabulary detection
    detections = model.open_vocabulary_detect(images, ["person", "car", "dog"])
    # detections[i]["boxes"], ["scores"], ["labels"]
    """
    print(usage)


def main():
    """Main function to generate model summary."""
    parser = argparse.ArgumentParser(description='KDEOV Model Summary and Visualization')
    parser.add_argument('--backbone', type=str, default='yolov8n',
                        choices=['yolov8n', 'yolov5s', 'simple'],
                        help='Backbone type (default: yolov8n)')
    parser.add_argument('--fusion', type=str, default='film',
                        choices=['film', 'cross_attention'],
                        help='Fusion type (default: film)')
    parser.add_argument('--static', action='store_true',
                        help='Show static summary only (no model loading)')
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print(" KDEOV Model Summary & Visualization".center(80))
    print("=" * 80)
    
    if args.static or not MODELS_AVAILABLE:
        if args.static:
            print("\n[INFO] Showing static summary (--static flag)")
        else:
            print("\n[WARNING] Cannot load model: Required dependencies are missing.")
            print("\nPlease install dependencies:")
            print("  pip install clip-by-openai")
            print("  pip install ultralytics  # Optional, for YOLO backbone")
        print("\n" + "=" * 80)
        print_static_summary()
        return
    
    # Weights directory: YOLO .pt and related weights are saved here
    weights_dir = os.path.join(os.getcwd(), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    # torch.hub (e.g. yolov5s) uses TORCH_HOME; point it under weights/
    os.environ["TORCH_HOME"] = os.path.join(weights_dir, "torch_hub")
    print(f"\n[OK] Weights directory: {weights_dir}")
    
    # Get device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n[OK] Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n[OK] Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print(f"\n[WARNING] Using CPU (slower, but will work)")
    
    # Initialize model
    print(f"\n[INIT] Initializing KDEOV Model...")
    print(f"  Backbone: {args.backbone}")
    print(f"  Fusion: {args.fusion}")
    try:
        model = KDEOVModel(
            clip_model_name="ViT-B/32",
            backbone_type=args.backbone,
            fusion_type=args.fusion,
            embedding_dim=512,
            weights_dir=weights_dir
        )
        print("[OK] Model initialized successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to initialize model: {e}")
        if args.backbone != "simple":
            print("\nTrying with fallback backbone...")
            try:
                model = KDEOVModel(
                    clip_model_name="ViT-B/32",
                    backbone_type="simple",  # Will use fallback
                    fusion_type=args.fusion,
                    embedding_dim=512,
                    weights_dir=weights_dir
                )
                print("[OK] Model initialized with fallback backbone!")
            except Exception as e2:
                print(f"[ERROR] Failed to initialize model: {e2}")
                print("\n[WARNING] Cannot proceed without model initialization.")
                print("   This may be due to missing CLIP weights or YOLO dependencies.")
                return
        else:
            print("\n[WARNING] Cannot proceed without model initialization.")
            print("   This may be due to missing CLIP weights.")
            return
    
    # Print architecture diagram
    print_architecture_diagram(model)
    
    # Print detailed summary
    print_detailed_summary(model, device)
    
    # Print comparison
    print_model_comparison()
    
    print("\n" + "=" * 80)
    print(" Summary Complete! ".center(80))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
