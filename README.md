# Knowledge Distillation for Efficient Open-Vocabulary Vision (KDEOV)
## Transferring CLIP's Semantic Alignment to Lightweight Models

## Group(CY2502) Members
- **PENG, Minqi** (1155191548)
- **ZHU, Keyu** (1155191834)

## Project Overview

This Final Year Project (FYP) focuses on knowledge distillation techniques for efficient open-vocabulary vision systems. The main objective is to transfer CLIP's semantic alignment capabilities to lightweight models, enabling efficient open-vocabulary visual recognition while maintaining high performance.

### Key Research Areas
- Knowledge distillation from large-scale vision-language models (CLIP)
- Efficient open-vocabulary vision systems
- Lightweight model optimization
- Semantic alignment transfer

### Current Model Structure (High Level)

The KDEOV model has a **text stream** and a **visual stream** with two output paths:

- **Text stream**: Frozen CLIP Text Encoder → text embeddings.
- **Visual stream**: Lightweight Visual Backbone (YOLOv8n/YOLOv5s or simple CNN) → multi-scale features → optional Cross-Modal Fusion (FiLM or cross-attention) with text. Then:
  - **Classification / retrieval path**: Projection Network (global pooling) → image embeddings; used for zero-shot classification and text-image retrieval.
  - **Detection path**: Spatial Projection (no pooling) → per-location embeddings; similarity with class-name text + grid default boxes + NMS → open-vocabulary object detection (boxes, scores, labels).

See **[models/README.md](./models/README.md)** for detailed architecture and API.

## Installation

## Data Preparation (Mini Dataset)

To ensure consistency across development environments (Mac/Linux/Windows), we use the **COCO128** dataset (a small 128-image subset of COCO) for testing and debugging.

**Do not download data manually.** Instead, run the automated script:

```bash
# This script will create a 'datasets/' folder and download COCO128 automatically
python download_data.py
```

## Environment Setup

Follow these steps to set up the development environment:

1. **Create a conda virtual environment with Python 3.9:**
   ```bash
   conda create -n KDEOV python=3.9
   conda activate KDEOV
   ```

2. **Install PyTorch:**
   
   **Important**: Choose the installation method based on your GPU's CUDA version. Check your CUDA version first:
   ```bash
   nvidia-smi  # Check CUDA version in the top right corner
   ```
   
   **For CUDA (GPU support):**
   - For CUDA 11.8:
     ```bash
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
     ```
   - For CUDA 12.1:
     ```bash
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
     ```
   - For CUDA 12.6:
     ```bash
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
     ```
   - Visit [PyTorch official website](https://pytorch.org/get-started/locally/) for other CUDA versions
   
   **For CPU only:**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```
   
   **Or using conda:**
   ```bash
   conda install pytorch torchvision -c pytorch
   ```

3. **Install other dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install YOLOv8 (if ultralytics installation fails):**
   
   If installing `ultralytics` via pip fails, you can use the alternative method:
   ```bash
   pip install git+https://github.com/ultralytics/yolov8.git
   ```

5. **Install CLIP:**
   ```bash
   pip install git+https://github.com/openai/CLIP.git
   ```

## Current Progress

### Term 1 (Completed)
The first term has been primarily dedicated to **theoretical research** and foundational work:

- **Literature Review**: Comprehensive review of existing knowledge distillation methods and open-vocabulary vision approaches
- **Theoretical Framework**: Development of theoretical foundations for transferring CLIP's semantic alignment to lightweight models
- **Problem Analysis**: Analysis of challenges and opportunities in efficient open-vocabulary vision
- **Research Proposal**: Detailed research proposal outlining the project scope, methodology, and expected contributions

### Winter Vocations - Implementation Phase

**Model Implementation (Completed):**
- ✅ **Core Model Components**: Implemented all key components of the KDEOV architecture
  - Frozen CLIP Text Encoder (`FrozenCLIPTextEncoder`)
  - Lightweight Visual Backbone (`LightweightVisualBackbone`) with YOLOv8n/YOLOv5s support
  - Projection Network (`ProjectionNetwork`) for image-level feature alignment (classification/retrieval path)
  - Cross-Modal Fusion Module (`CrossModalFusionModule`) with FiLM and Cross-Attention support
  - Spatial Projection (`SpatialProjection`) for per-location embeddings (open-vocabulary detection path)
- ✅ **Dual-path visual output**: Shared backbone + fusion, then (1) Projection Network → image embeddings for zero-shot classification and retrieval; (2) Spatial Projection → spatial embeddings for open-vocabulary object detection (boxes, scores, labels).
- ✅ **Loss Functions**: Implemented comprehensive loss functions
  - Distillation Loss (Cosine and L2 variants)
  - Cross-Modal Alignment Loss (InfoNCE-based)
  - Feature Alignment Loss (combined loss for end-to-end training)
- ✅ **Main Model**: Complete `KDEOVModel` class with full training and inference interface, including `open_vocabulary_detect()` and `get_spatial_embeddings()`.
- ✅ **Training Script**: Feature alignment pretraining script (`train_feature_alignment.py`)
- ✅ **Usage Examples**: Zero-shot classification, text-image retrieval, forward pass, and open-vocabulary detection (see Usage Guide and `models/README.md`)

**Environment Setup (Completed):**
- ✅ Conda environment configuration (Python 3.9)
- ✅ PyTorch installation with CUDA support
- ✅ All dependencies installed and verified
- ✅ IDE configuration for development

**Next Steps:**
- Dataset preparation and data loading implementation
- Model training on real datasets
- Performance evaluation and benchmarking
- Model optimization and fine-tuning

### Documentation

This repository contains comprehensive documentation organized into the following categories:

#### Project Documentation
- **[README.md](./README.md)** (this file)
  - **Purpose**: Primary project documentation providing project overview, installation instructions, usage guidelines, and quick start guide
  - **Contents**: Project introduction, environment configuration, code usage guide, and documentation index
  - **Target Audience**: All users (newcomers for getting started, developers for reference)

- **[Development_Log.md](./Development_Log.md)**
  - **Purpose**: Development log and work records
  - **Contents**: Detailed daily work records, environment configuration procedures, code implementation details, encountered issues and their solutions
  - **Target Audience**: Project developers and maintainers for tracking development progress and troubleshooting

#### Model Documentation
- **[models/README.md](./models/README.md)**
  - **Purpose**: Model architecture and technical documentation
  - **Contents**: Detailed model component descriptions, loss function introductions, API documentation, and model design principles
  - **Target Audience**: Researchers and developers requiring in-depth understanding of the model architecture

#### Academic Documentation
- **[Documents/FYP Research Proposal.pdf](./Documents/FYP%20Research%20Proposal.pdf)**
  - **Purpose**: Research proposal articulating project research objectives, methodology, and expected contributions
  - **Target Audience**: Academic reviewers, project supervisors, and researchers

- **[Documents/FYP Term1 Midterm Report.pdf](./Documents/FYP%20Term1%20Midterm%20Report.pdf)**
  - **Purpose**: First term midterm report documenting theoretical research and preliminary progress
  - **Target Audience**: Academic reviewers and project supervisors

- **[Documents/FYP Term1 End Report.pdf](./Documents/FYP%20Term1%20End%20Report.pdf)**
  - **Purpose**: First term final report summarizing research achievements during the theoretical phase
  - **Target Audience**: Academic reviewers and project supervisors

## Python Files Documentation and Usage Guide

### Core Model Files (`models/`)

#### `models/__init__.py`
- **Purpose**: Package initialization file that exports all public APIs
- **Usage**: Import model components via `from models import ...`
- **Exported Components**:
  - `KDEOVModel` - Main model class
  - `FrozenCLIPTextEncoder` - Frozen CLIP text encoder
  - `LightweightVisualBackbone` - Lightweight visual backbone network
  - `ProjectionNetwork` - Projection network (classification/retrieval path)
  - `CrossModalFusionModule` - Cross-modal fusion module
  - `SpatialProjection` - Per-location projection (detection path); `grid_boxes_to_image` - detection utility
  - `DistillationLoss`, `CrossModalAlignmentLoss`, `FeatureAlignmentLoss` - Loss functions

#### `models/components.py`
- **Purpose**: Implementation of all model components
- **Components**:
  - `FrozenCLIPTextEncoder` - Utilizes pretrained CLIP text encoder
  - `LightweightVisualBackbone` - YOLOv8n/YOLOv5s visual backbone network
  - `ProjectionNetwork` - Feature projection with global pooling (image-level embeddings)
  - `CrossModalFusionModule` - FiLM or Cross-Attention fusion module
  - `SpatialProjection` - 1×1 conv + GroupNorm, preserves spatial dims for detection; `grid_boxes_to_image` for default boxes
- **Usage**: Typically used indirectly through `KDEOVModel`, but can be imported separately for custom model implementations

#### `models/kdeov_model.py`
- **Purpose**: Main model class integrating all components; dual-path output (classification/retrieval and detection).
- **Key Functionalities**:
  - Model initialization and configuration
  - Image and text encoding (`encode_image`, `encode_text`)
  - Zero-shot classification (`zero_shot_classify`)
  - Text-image retrieval (similarity via `compute_similarity`)
  - Open-vocabulary object detection (`open_vocabulary_detect`) — returns boxes, scores, labels per image
  - Spatial embeddings for detection (`get_spatial_embeddings`)
  - Forward propagation and feature extraction
- **Usage**:
  ```python
  from models import KDEOVModel
  
  model = KDEOVModel(
      clip_model_name="ViT-B/32",  # CLIP model name
      backbone_type="yolov8n",     # Visual backbone: yolov8n or yolov5s
      fusion_type="film"            # Fusion method: film or cross_attention
  ).cuda()
  ```

#### `models/losses.py`
- **Purpose**: Implementation of all loss functions
- **Loss Functions**:
  - `DistillationLoss` - Knowledge distillation loss (Cosine or L2 variants)
  - `CrossModalAlignmentLoss` - Cross-modal alignment loss (InfoNCE-based)
  - `FeatureAlignmentLoss` - Combined loss function
- **Usage**:
  ```python
  from models import FeatureAlignmentLoss
  
  criterion = FeatureAlignmentLoss(
      distillation_weight=1.0,    # Distillation loss weight
      alignment_weight=1.0,      # Alignment loss weight
      distillation_type="cosine", # Distillation type: cosine or l2
      temperature=0.07             # Temperature parameter
  )
  ```

### Training and Example Files

#### `train_feature_alignment.py`
- **Purpose**: Feature alignment pretraining script
- **Functionalities**:
  - Complete training loop implementation
  - Support for distillation loss and cross-modal alignment loss
  - Automatic checkpoint saving
  - Learning rate scheduling and gradient clipping
- **Usage Methods**:
  ```bash
  # Method 1: Direct execution (using built-in example data)
  python train_feature_alignment.py
  
  # Method 2: Import and call in code
  from train_feature_alignment import train_feature_alignment
  from models import KDEOVModel
  from torch.utils.data import DataLoader
  
  # Initialize model
  model = KDEOVModel(...).cuda()
  
  # Prepare data loader (requires custom dataset implementation)
  dataloader = DataLoader(your_dataset, batch_size=32, shuffle=True)
  
  # Start training
  train_feature_alignment(
      model=model,
      dataloader=dataloader,
      num_epochs=10,
      learning_rate=1e-4,
      save_path="checkpoints/kdeov"  # Checkpoint save path
  )
  ```
- **Parameter Specifications**:
  - `num_epochs`: Number of training epochs (default: 10)
  - `learning_rate`: Learning rate (default: 1e-4)
  - `save_path`: Checkpoint save path (optional)

#### `example_usage.py`
- **Purpose**: Model usage examples and demonstrations
- **Included Examples**:
  1. **Zero-shot Classification** (`example_zero_shot_classification`)
     - Demonstrates zero-shot image classification using the model
  2. **Text-Image Retrieval** (`example_text_image_retrieval`)
     - Demonstrates retrieving most similar images based on text queries
  3. **Forward Pass** (`example_forward_pass`)
     - Demonstrates standard model forward propagation
- **Usage Methods**:
  ```bash
  # Run all examples
  python example_usage.py
  
  # Or call individual examples in code
  from example_usage import example_zero_shot_classification
  example_zero_shot_classification()
  ```
- **Important Notes**:
  - Examples use random data; replace with real images and text for actual usage
  - First run will automatically download CLIP pretrained weights

### Utility Scripts

#### `test_scripts/test_environment.py`
- **Purpose**: Comprehensive environment verification script for CUDA, CLIP, and ultralytics
- **Functionality**:
  - Verifies CUDA availability and GPU information
  - Tests CLIP import and basic functionality (lists available models without downloading)
  - Tests ultralytics import and YOLO class availability
  - Performs integration tests with CUDA tensor operations
- **Usage** (run from project root):
  ```bash
  python test_scripts/test_environment.py
  ```
- **Use Cases**:
  - Complete environment check: Verifies all required components (CUDA, CLIP, ultralytics) in one script
  - GPU information: Displays detailed GPU specifications and memory
  - Quick verification: Run before starting training to ensure everything is properly configured
  - No downloads: The script only checks imports and basic functionality without downloading models

#### `test_scripts/test_backbone.py`
- **Purpose**: Test script for verifying the `LightweightVisualBackbone` component functionality
- **Functionality**:
  - Tests import of `LightweightVisualBackbone` from `models.components`
  - Initializes YOLOv8n backbone model (automatically downloads pretrained weights if needed)
  - Performs forward pass with dummy input tensor (batch=1, channels=3, height=640, width=640)
  - Validates output feature shapes and structure
  - Provides detailed error messages for debugging backbone issues
- **Usage** (run from project root):
  ```bash
  python test_scripts/test_backbone.py
  ```
- **Use Cases**:
  - Backbone verification: Ensures the YOLO backbone is correctly integrated and functional
  - Debugging: Helps identify issues with backbone initialization or forward pass
  - Feature shape validation: Verifies that the backbone outputs expected multi-scale features
  - First-time setup: Useful for confirming YOLO weights download and model loading
- **Important Notes**:
  - First run will automatically download YOLOv8n pretrained weights (`yolov8n.pt`)
  - The script uses dummy random tensors; replace with real images for actual feature extraction testing
  - Outputs detailed step-by-step progress and error information for troubleshooting

#### `test_scripts/model_summary.py`
- **Purpose**: Comprehensive model summary and visualization script for KDEOV model architecture analysis
- **Functionality**:
  - **Model Architecture Visualization**: Displays text-based architecture diagram showing data flow through all components, including the dual-path design (Projection Network for classification/retrieval, Spatial Projection for open-vocabulary detection)
  - **Parameter Statistics**: Counts and displays total, trainable, and frozen parameters for the entire model and each component (including Spatial Projection)
  - **Component Analysis**: Provides detailed breakdown of each model component (Text Encoder, Visual Backbone, Projection Network, Fusion Module, Spatial Projection)
  - **Input/Output Shape Testing**: Automatically tests forward pass and captures input/output tensor shapes (image/text embeddings, zero-shot logits, spatial embeddings, and detection boxes/scores/labels when available)
  - **Memory Usage Estimation**: Calculates model size in MB (assuming float32 parameters)
  - **Training Information**: Lists trainable vs frozen components and available loss functions
  - **Model Comparison**: Compares KDEOV model size with full CLIP model
  - **Static Summary Mode**: Works even without CLIP installed, showing architecture overview and both visual output paths
- **Usage** (run from project root):
  ```bash
  # Basic usage (default: yolov8n backbone, film fusion)
  python test_scripts/model_summary.py

  # With specific backbone and fusion type
  python test_scripts/model_summary.py --backbone yolov5s --fusion cross_attention

  # Static summary only (no model loading, works without dependencies)
  python test_scripts/model_summary.py --static
  ```
- **Command-line Options**:
  - `--backbone BACKBONE`: Choose backbone type (`yolov8n`, `yolov5s`, or `simple`)
  - `--fusion FUSION`: Choose fusion type (`film` or `cross_attention`)
  - `--static`: Show static summary only (no model loading, useful when dependencies are missing)
- **Use Cases**:
  - **Model Inspection**: Quickly understand model architecture and parameter distribution
  - **Debugging**: Verify model structure, parameter counts, and component integration
  - **Documentation**: Generate model statistics for reports and presentations
  - **Performance Analysis**: Check trainable vs frozen parameters to understand training efficiency
  - **Environment Verification**: Test if model can be initialized correctly with current setup
  - **Architecture Understanding**: Visualize data flow and component relationships
- **Output Sections**:
  1. **Model Architecture Diagram**: Visual representation of text and visual streams and dual-path output (Projection Network vs Spatial Projection)
  2. **Overall Statistics**: Total/trainable/frozen parameters, model size, device info
  3. **Component Details**: Individual component parameter counts (including Spatial Projection) and descriptions
  4. **Input/Output Shapes**: Verified tensor shapes for image/text embeddings, zero-shot logits, spatial embeddings, and detection outputs
  5. **Training Information**: Trainable components and loss function descriptions
  6. **Model Size Comparison**: Comparison with full CLIP model
  7. **Usage Examples**: Quick code snippets for model usage (including open-vocabulary detection)
- **Important Notes**:
  - First run will automatically download CLIP and YOLO pretrained weights if needed
  - The script handles missing dependencies gracefully (shows static summary if CLIP is not installed)
  - Forward pass testing requires CUDA/CLIP to be available
  - Parameter counts are accurate and include all submodules
  - Model size calculation assumes float32 (4 bytes per parameter)

### Configuration Files

#### `requirements.txt`
- **Purpose**: Python dependency package list
- **Usage**:
  ```bash
  pip install -r requirements.txt
  ```
- **Dependencies**: numpy, Pillow, ftfy, regex, tqdm, ultralytics (YOLOv8)

## Usage Guide

### Quick Start

#### 1. Verify Environment Configuration

First, ensure the conda environment is activated and all dependencies are installed:

```bash
conda activate KDEOV
```

**Verify dependencies installation:**

You can verify that required packages are properly installed by running:

```bash
# Method 1: Comprehensive verification (recommended)
python test_scripts/test_environment.py

# Method 2: Quick verification
python -c "import torch; import clip; from ultralytics import YOLO; print('Environment OK!')"
```

The `test_scripts/test_environment.py` script provides comprehensive verification:
- Confirms CUDA availability and displays GPU information
- Verifies that imports work successfully (`import clip` and `import ultralytics`)
- Lists available CLIP models without downloading them
- Tests basic CUDA tensor operations
- Provides immediate feedback if packages are not installed correctly
- Displays helpful installation instructions if packages are missing

#### 2. Verify Model Structure (Optional but Recommended)

```bash
# Generate comprehensive model summary and statistics
python test_scripts/model_summary.py
```

This will display:
- Model architecture diagram
- Parameter counts (total, trainable, frozen)
- Component details and descriptions
- Input/output tensor shapes
- Training information
- Model size comparison with CLIP

#### 3. Run Example Code

```bash
# Run usage examples (to understand model API)
python example_usage.py
```

#### 4. Initialize Model

```python
import torch
from models import KDEOVModel

# Create model instance (automatically on CUDA)
model = KDEOVModel(
    clip_model_name="ViT-B/32",  # Options: RN50, RN101, ViT-B/16, ViT-L/14, etc.
    backbone_type="yolov8n",    # Options: yolov8n, yolov5s
    fusion_type="film"           # Options: film, cross_attention
).cuda()

# Set to evaluation mode
model.eval()
```

#### 5. Zero-Shot Classification

```python
import clip
from PIL import Image
import torchvision.transforms as transforms

# Load and preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711))
])

image = Image.open("your_image.jpg")
image_tensor = transform(image).unsqueeze(0).cuda()

# Define class names
class_names = ["cat", "dog", "bird", "car", "bicycle"]

# Perform classification
with torch.no_grad():
    logits = model.zero_shot_classify(image_tensor, class_names)
    probs = torch.softmax(logits, dim=-1)
    predicted_idx = torch.argmax(probs, dim=-1).item()
    
print(f"Predicted: {class_names[predicted_idx]} (confidence: {probs[0][predicted_idx]:.4f})")
```

#### 6. Text-Image Retrieval

```python
import clip

# Prepare image database (example)
image_database = [...]  # Your list of images
image_tensors = torch.stack([transform(img) for img in image_database]).cuda()

# Query text
query_text = "a photo of a cat"
text_tokens = clip.tokenize([query_text]).cuda()

# Encode
with torch.no_grad():
    image_embeddings = model.encode_image(image_tensors)
    text_embeddings = model.encode_text(text_tokens)

# Compute similarity and retrieve
similarities = model.compute_similarity(image_embeddings, text_embeddings)
top_k_indices = torch.topk(similarities, k=5, dim=0).indices

print(f"Top 5 most similar images for '{query_text}':")
for i, idx in enumerate(top_k_indices):
    print(f"  Rank {i+1}: Image {idx.item()} (similarity: {similarities[idx].item():.4f})")
```

#### 7. Open-Vocabulary Object Detection

```python
# Same image preprocessing as above (e.g. transform, image_tensor [1, 3, 224, 224])
# Define open-vocabulary class names (any text labels)
class_names = ["person", "car", "dog", "bus", "bicycle"]

with torch.no_grad():
    detections = model.open_vocabulary_detect(
        image_tensor,
        class_names=class_names,
        score_threshold=0.2,
        nms_threshold=0.5,
        max_detections_per_image=50,
    )

# detections is a list of dicts (one per image)
for i, det in enumerate(detections):
    boxes = det["boxes"]   # [N, 4] in xyxy format (x1, y1, x2, y2)
    scores = det["scores"] # [N]
    labels = det["labels"] # list of N class name strings
    print(f"Image {i}: {len(labels)} detections")
    for box, score, label in zip(boxes, scores, labels):
        print(f"  {label}: {score:.2f} @ {box.tolist()}")
```

#### 8. Model Training

```python
from train_feature_alignment import train_feature_alignment
from torch.utils.data import Dataset, DataLoader

# Implement your dataset class
class YourDataset(Dataset):
    def __init__(self):
        # Initialize dataset
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return (image_tensor, text_tokens) pair
        return image, text_tokens

# Create data loader
dataset = YourDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model (automatically on CUDA)
model = KDEOVModel(...).cuda()

# Start training
train_feature_alignment(
    model=model,
    dataloader=dataloader,
    num_epochs=10,
    learning_rate=1e-4,
    save_path="checkpoints/kdeov"
)
```

### Model Execution Workflow

1. **Environment Preparation**: Activate conda environment and ensure all dependencies are installed
2. **Model Initialization**: Create `KDEOVModel` instance with selected configuration parameters
3. **Data Preparation**: Prepare image and text data (for training or inference)
4. **Model Inference**: Call appropriate methods:
   - **Zero-shot classification**: `zero_shot_classify(images, class_names)`
   - **Text-image retrieval**: `encode_image`, `encode_text`, `compute_similarity`
   - **Open-vocabulary detection**: `open_vocabulary_detect(images, class_names, ...)` → boxes, scores, labels
5. **Model Training** (Optional): Use training script for feature alignment pretraining

## Repository Structure

```
KDEOV/
├── models/                           # Model implementation directory
│   ├── __init__.py                  # Package initialization file, exports all public APIs
│   ├── components.py                 # Model component implementations (encoders, backbones, fusion modules)
│   ├── kdeov_model.py               # Main model class KDEOVModel
│   ├── losses.py                     # Loss function implementations (distillation loss, alignment loss, etc.)
│   └── README.md                     # Detailed model architecture documentation
│
├── train_feature_alignment.py        # Feature alignment pretraining script
│                                      # Purpose: Train KDEOV model
│                                      # Execution: python train_feature_alignment.py
│
├── example_usage.py                  # Model usage examples
│                                      # Purpose: Demonstrate zero-shot classification, text-image retrieval, etc.
│                                      # Execution: python example_usage.py
│
├── test_scripts/                     # Test and utility scripts
│   ├── test_environment.py           # Environment verification (CUDA, CLIP, ultralytics)
│   │                                    # Execution: python test_scripts/test_environment.py
│   ├── test_backbone.py              # YOLO backbone testing
│   │                                    # Execution: python test_scripts/test_backbone.py
│   └── model_summary.py              # Model summary and visualization
│                                        # Execution: python test_scripts/model_summary.py [--backbone BACKBONE] [--fusion FUSION] [--static]
│
├── requirements.txt                  # Python dependency package list
│                                      # Purpose: Define all required Python packages for the project
│                                      # Usage: pip install -r requirements.txt
│
├── README.md                         # Main project documentation (this file)
│                                      # Purpose: Project overview, installation guide, usage instructions
│
├── Development_Log.md                # Development log
│                                      # Purpose: Record development process, work content, problem resolution
│
├── Documents/                        # Academic documentation directory
│   ├── FYP Research Proposal.pdf    # Research proposal
│   ├── FYP Term1 Midterm Report.pdf # First term midterm report
│   └── FYP Term1 End Report.pdf     # First term final report
│
└── .vscode/                          # IDE configuration directory
    └── settings.json                 # VS Code/Cursor Python interpreter configuration
                                        # Purpose: Specify conda environment path, resolve import issues
```

### File Function Quick Reference

| File/Directory | Type | Primary Function | Execution Method |
|---------------|------|-----------------|------------------|
| `models/__init__.py` | Python Module | Export model APIs | `from models import ...` |
| `models/components.py` | Python Module | Model component implementations | Used via `KDEOVModel` |
| `models/kdeov_model.py` | Python Module | Main model class | `from models import KDEOVModel` |
| `models/losses.py` | Python Module | Loss functions | `from models import FeatureAlignmentLoss` |
| `train_feature_alignment.py` | Executable Script | Model training | `python train_feature_alignment.py` |
| `example_usage.py` | Executable Script | Usage examples | `python example_usage.py` |
| `test_scripts/test_environment.py` | Executable Script | Verify CUDA, CLIP, ultralytics | `python test_scripts/test_environment.py` |
| `test_scripts/test_backbone.py` | Executable Script | Test YOLO backbone functionality | `python test_scripts/test_backbone.py` |
| `test_scripts/model_summary.py` | Executable Script | Model architecture analysis and statistics | `python test_scripts/model_summary.py` |
| `requirements.txt` | Configuration File | Dependency management | `pip install -r requirements.txt` |
| `README.md` | Documentation | Project description | Reference reading |
| `Development_Log.md` | Documentation | Development records | Reference reading |
| `models/README.md` | Documentation | Model documentation | Reference reading |