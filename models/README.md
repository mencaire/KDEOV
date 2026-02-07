# KDEOV Model Architecture

This directory contains the implementation of the Knowledge Distillation for Efficient Open-Vocabulary Vision (KDEOV) model based on Section 3 Methodology.

**Current model structure (high level):**
- **Text stream**: Frozen CLIP Text Encoder → text embeddings.
- **Visual stream**: Lightweight Visual Backbone (YOLOv8n / YOLOv5s or simple CNN) → multi-scale features → optional Cross-Modal Fusion (FiLM or cross-attention) with text. Then:
  - **Classification / retrieval**: Projection Network (global pooling) → image embeddings; similarity with text for zero-shot classification and retrieval.
  - **Open-vocabulary detection**: Spatial Projection (no pooling) → per-location embeddings; similarity with class-name text + grid boxes + NMS → boxes, scores, labels.

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

### 3. Projection Network (`ProjectionNetwork`) & Global Alignment
- **Structure**: 2-layer MLP that maps image features to CLIP embedding space.
- **Pooling**: Applies **Global Average Pooling (GAP)** to convert spatial feature maps `[B, C, H, W]` into global vectors `[B, C]`.
- **Normalization**: Applies **L2 Normalization** to ensure embeddings lie on a hypersphere.
- **Output**: Normalized global image embeddings ready for cosine similarity calculation with text.


### 4. Cross-Modal Fusion Module (`CrossModalFusionModule`)
- Implements Feature-wise Linear Modulation (FiLM) or cross-attention
- Fuses text embeddings with image features
- Enables text-guided visual processing
- Supports two fusion types:
  - **FiLM**: Feature-wise Linear Modulation (scale and shift)
  - **Cross-Attention**: Attention-based fusion

### 5. Spatial Projection (`SpatialProjection`)
- 1×1 conv + GroupNorm that maps backbone feature maps to CLIP embedding space
- **Preserves spatial dimensions** (no global pooling): output shape `[B, embedding_dim, Hf, Wf]`
- Used for open-vocabulary detection: each spatial location has an embedding for region–text similarity
- Shared backbone and optional fusion; then spatial projection (detection path) vs. Projection Network (classification path)

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
Integrates all components into a unified model.

**Key Features:**
- **Learnable Temperature**: Includes a learnable `logit_scale` parameter (initialized to $\log(1/0.07)$) to scale cosine similarities, critical for InfoNCE loss convergence.
- **Dual-Path Output**:
  - **Global Path**: Returns normalized vectors for contrastive learning.
  - **Spatial Path**: Returns feature maps for detection.

**Forward Output (`forward` returns a Dictionary):**
- `visual_features`: Normalized global image embeddings `[B, 512]` (for contrastive loss).
- `text_features`: Normalized text embeddings `[B, 512]` (if text provided).
- `logits`: Scaled similarity scores `[B, B]` calculated as $(I \cdot T) \times \exp(\text{logit\_scale})$.
- `visual_map`: The spatial feature map before pooling (useful for debugging).
- `fused_map`: (Optional) Feature map after text-visual fusion.
- `predictions`: (Optional) Detection head outputs.


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

# Open-vocabulary object detection
detections = model.open_vocabulary_detect(
    images,
    class_names=["person", "car", "dog"],
    score_threshold=0.2,
    nms_threshold=0.5,
)
# detections[i]["boxes"], ["scores"], ["labels"] per image
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

### Model Structure Overview

The model has **two visual output paths** sharing the same backbone and fusion:

| Path | Use | After backbone + fusion | Output |
|------|-----|-------------------------|--------|
| **Classification / retrieval** | `encode_image`, `forward`, `zero_shot_classify` | Projection Network (global pooling) | Image embeddings `[B, 512]` |
| **Detection** | `get_spatial_embeddings`, `open_vocabulary_detect` | Spatial Projection (no pooling) | Per-location embeddings `[B, 512, Hf, Wf]` |

- **Text stream**: Text → Frozen CLIP Text Encoder → Text Embeddings `[B, 512]`.
- **Visual stream**: Image → Lightweight Backbone → multi-scale features → (optional) Fusion with text → either Projection Network or Spatial Projection.

### Inference Flow

1. **Image-level (classification / retrieval)**:
   - Images → Lightweight Backbone → Multi-scale Features
   - Features → Projection Network → Spatial Map `[B, 512, H, W]`
   - Spatial Map → **Global Average Pooling** → **L2 Normalization** → Image Embeddings `[B, 512]`
   - **Similarity Calculation**: Dot product between normalized Image and Text embeddings, scaled by `logit_scale`.


2. **Open-vocabulary detection**:
   - Images → Lightweight Backbone → Multi-scale Features
   - Features → Fusion Module (with text) → Fused Features
   - Fused Features → **Spatial Projection** (no pooling) → Per-location embeddings `[B, 512, Hf, Wf]`
   - Text → Frozen CLIP Text Encoder → Class name embeddings
   - Similarity at each location with each class → scores; grid default boxes → NMS → detections (boxes, scores, labels)

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

