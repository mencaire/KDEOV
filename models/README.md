# KDEOV Model Architecture

This directory contains the implementation of the Knowledge Distillation for Efficient Open-Vocabulary Vision (KDEOV) model based on Section 3 Methodology.

## Model Components

### 1. Frozen CLIP Text Encoder (`FrozenCLIPTextEncoder`)
- Uses pretrained CLIP text encoder as a frozen semantic reference
- Processes text prompts and outputs semantic embeddings in CLIP embedding space
- All parameters are frozen (no gradients)

### 2. Lightweight Visual Backbone (`LightweightVisualBackbone`)
- Uses YOLO backbone (YOLOv5s or YOLOv8n) for efficient feature extraction
- Extracts multi-scale features suitable for object-level representation
- Supports both YOLOv8n and YOLOv5s backbones
- Falls back to simple CNN if YOLO is not available

### 3. Projection Network (`ProjectionNetwork`)
- 2-layer MLP that maps image features to CLIP embedding space
- Includes LayerNorm and dropout for regularization
- Outputs normalized embeddings aligned with CLIP space

### 4. Cross-Modal Fusion Module (`CrossModalFusionModule`)
- Implements Feature-wise Linear Modulation (FiLM) or cross-attention
- Fuses text embeddings with image features
- Enables text-guided visual processing
- Supports two fusion types:
  - **FiLM**: Feature-wise Linear Modulation (scale and shift)
  - **Cross-Attention**: Attention-based fusion

## Loss Functions

### 1. Distillation Loss (`DistillationLoss`)
- Transfers semantic richness from large CLIP to compact student model
- Supports two loss types:
  - **Cosine**: Cosine similarity loss (1 - cosine_sim)
  - **L2**: Mean squared error loss
- Aligns student image embeddings with teacher CLIP embeddings

### 2. Cross-Modal Alignment Loss (`CrossModalAlignmentLoss`)
- Uses contrastive loss (InfoNCE) for semantic alignment
- Ensures image and text embeddings are aligned in shared embedding space
- Maximizes similarity for matched pairs, minimizes for mismatches

### 3. Feature Alignment Loss (`FeatureAlignmentLoss`)
- Combined loss for end-to-end training
- Combines distillation loss and cross-modal alignment loss
- Configurable weights for each component

## Main Model

### KDEOVModel
Integrates all components into a unified model that supports:
- **Zero-shot classification**: Classify images using text prompts
- **Text-image retrieval**: Find images matching text queries
- **Feature extraction**: Extract aligned image and text embeddings

## Usage

### Basic Usage

```python
from models import KDEOVModel
import torch
import clip

# Initialize model (automatically on CUDA)
model = KDEOVModel(
    clip_model_name="ViT-B/32",
    backbone_type="yolov8n",
    fusion_type="film"
).cuda()

# Encode images
images = torch.randn(4, 3, 224, 224).cuda()
image_embeddings = model.encode_image(images)

# Encode text
text = clip.tokenize(["a photo of a cat"]).cuda()
text_embeddings = model.encode_text(text)

# Zero-shot classification
class_names = ["cat", "dog", "bird"]
logits = model.zero_shot_classify(images, class_names)
```
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>
read_file

### Training

See `train_feature_alignment.py` for a complete training example.

```python
from models import KDEOVModel, FeatureAlignmentLoss
from train_feature_alignment import train_feature_alignment

model = KDEOVModel(...)
# ... setup dataloader ...

train_feature_alignment(
    model=model,
    dataloader=dataloader,
    num_epochs=10,
    learning_rate=1e-4
)
```

## Architecture Details

### Feature Alignment Pretraining

During pretraining, the model learns to:
1. **Distill** semantic knowledge from frozen CLIP image encoder to lightweight backbone
2. **Align** image and text embeddings in shared semantic space
3. **Fuse** text and image features for text-guided processing

### Inference Flow

1. **Image Processing**:
   - Images → Lightweight Backbone → Multi-scale Features
   - Features → Fusion Module (with text) → Fused Features
   - Fused Features → Projection Network → Image Embeddings

2. **Text Processing**:
   - Text → Frozen CLIP Text Encoder → Text Embeddings

3. **Similarity Computation**:
   - Cosine similarity between image and text embeddings

## File Structure

```
models/
├── __init__.py              # Package exports
├── components.py            # Model components
├── losses.py                # Loss functions
├── kdeov_model.py          # Main model class
└── README.md               # This file
```

## Dependencies

- PyTorch >= 1.12.0
- torchvision >= 0.13.0
- clip-by-openai >= 1.0
- ultralytics >= 8.0.0 (for YOLOv8)

See `requirements.txt` for complete list.

