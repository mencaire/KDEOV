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

## Data Preparation

**Do not download data manually.** Use the automated script:

```bash
# COCO128 (quick test, 128 images)
python download_data.py --dataset coco128

# COCO2017 (full training)
python download_data.py --dataset coco2017 --parts train2017 val2017 annotations

# LVIS (OVOD benchmark, requires COCO2017 images)
python download_data.py --dataset lvis

# COCO + LVIS (recommended for Open-Vocabulary Object Detection)
python download_data.py --dataset coco_lvis
```

**Minimal workflow:** run COCO128 (sanity check), then download and train with `coco_lvis`. See **[TRAINING_GUIDE.md](./TRAINING_GUIDE.md)** for the full COCO + LVIS training workflow.

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

### Term 2 (Ongoing) — From February

**Dataset, Finetuning & Evaluation (Completed):**
- ✅ **Dataset preparation**: COCO + LVIS (`coco_lvis`) prepared for feature alignment and open-vocabulary detection
- ✅ **Finetuning**: Training pipeline on COCO 2017 for detection finetuning
- ✅ **Evaluation**: Evaluation pipeline and benchmarking in place

**Current issue:**
- **AP/AR = 0**: Detection evaluation currently reports zero Average Precision and Average Recall; debugging and fixing this is in progress

**Planned next steps:**
- **Resolve AP/AR = 0**: Diagnose evaluation, labels, or training
- **Model optimization**: Further fine-tuning once metrics are valid

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

These scripts follow the workflow in **[TRAINING_GUIDE.md](./TRAINING_GUIDE.md)**: Phase 1 (sanity check) → Phase 2 (feature alignment on COCO+LVIS) → Phase 3 (detection fine-tuning with bbox) → Phase 4 (evaluation on val2017).

#### `train_feature_alignment.py`
- **Purpose**: Feature alignment pretraining (Phase 1 and Phase 2). Trains the KDEOV model with distillation loss and cross-modal alignment loss so that visual and text embeddings align. No bounding box labels; uses image–caption or image–category pairs from the chosen dataset.
- **Functionalities**:
  - Training loop with distillation loss and cross-modal alignment loss
  - Automatic checkpoint saving (per-epoch and optional resume)
  - Supports datasets: `coco128` (sanity check), `coco2017`, `lvis`, `coco_lvis` (recommended for OVOD), `dummy`
- **Usage** (see [TRAINING_GUIDE.md](./TRAINING_GUIDE.md) Phase 1 and Phase 2):
  ```bash
  # Phase 1: Quick sanity check (COCO128)
  python train_feature_alignment.py --dataset coco128 --epochs 5 --batch-size 16

  # Phase 2: Full feature alignment (COCO + LVIS) — after: python download_data.py --dataset coco_lvis
  python train_feature_alignment.py --dataset coco_lvis --split train --epochs 10 --batch-size 32 --save-path checkpoints/kdeov_coco_lvis
  ```
- **Options**: `--dataset` (coco128 | coco2017 | lvis | coco_lvis | dummy), `--data-root`, `--split` (train | val), `--epochs`, `--lr`, `--batch-size`, `--save-path`, `--resume`, `--backbone`, `--fusion`. See [TRAINING_GUIDE.md](./TRAINING_GUIDE.md).

#### `train_detection_finetune.py`
- **Purpose**: Detection fine-tuning (Phase 3 / Step 3). After feature alignment, fine-tune with **bounding box labels** from COCO train2017 (`instances_train2017.json`). The grid cell responsible for each GT box (center or best-IoU default box) is trained to predict the correct class; optional bbox regression (smooth L1 or GIoU) and negative (background) cell loss. No extra download if `coco_lvis` is already present.
- **Functionalities**:
  - Loads a feature-alignment checkpoint and trains on COCO 2017 detection data
  - Classification loss at the responsible cell (spatial feature · text embedding); optional regression and negative-cell loss
  - Saves one checkpoint (e.g. `kdeov_finetune.pt`) under `--save-path` for use in Phase 4
- **Usage** (see [TRAINING_GUIDE.md](./TRAINING_GUIDE.md) Phase 3):
  ```bash
  python train_detection_finetune.py --checkpoint checkpoints/kdeov_coco_lvis_epoch_10.pt --epochs 5 --batch-size 16 --save-path checkpoints/kdeov_finetune
  ```
- **Options**: `--checkpoint` (required), `--data-root`, `--save-path`, `--epochs`, `--batch-size`, `--lr`, `--backbone`, `--fusion`, `--reg-weight`, `--neg-weight`, `--neg-margin`, `--max-neg-per-image`, `--no-best-iou-cell`, `--reg-loss` (smooth_l1 | giou). See [TRAINING_GUIDE.md](./TRAINING_GUIDE.md) Phase 3.

#### `eval_detection.py`
- **Purpose**: Evaluation (Phase 4 / Step 4). Run the trained model on **val2017** and compute detection metrics. Use the **fine-tuned** checkpoint if you ran Phase 3; otherwise the feature-alignment checkpoint. Computes mAP, AP@50; on LVIS val also AP, AP_rare, AP_common, AP_frequent.
- **Usage** (see [TRAINING_GUIDE.md](./TRAINING_GUIDE.md) Phase 4):
  ```bash
  # LVIS val (primary OVOD benchmark) — use fine-tuned checkpoint when available
  python eval_detection.py --checkpoint checkpoints/kdeov_finetune/kdeov_finetune.pt --dataset lvis

  # COCO val (optional 80-class comparison)
  python eval_detection.py --checkpoint checkpoints/kdeov_finetune/kdeov_finetune.pt --dataset coco
  ```
  *(If you skipped Phase 3, use* `--checkpoint checkpoints/kdeov_coco_lvis_epoch_10.pt` *instead.)*
- **Options**: `--checkpoint` (required), `--data-root`, `--dataset` (lvis | coco), `--batch-size`, `--score-thresh` (default 0.01), `--backbone`, `--fusion`. See [TRAINING_GUIDE.md](./TRAINING_GUIDE.md) Phase 4.

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

#### 4. Initialize Model

#### 5. Zero-Shot Classification

#### 6. Text-Image Retrieval

#### 7. Open-Vocabulary Object Detection

#### 8. Model Training

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
├── data/                             # Data loading and dataset implementations
│   ├── __init__.py                  # Package initialization, exports dataset classes
│   ├── coco_dataset.py              # COCO 2017 dataset for detection / feature alignment
│   ├── detection_dataset.py         # Generic detection dataset and collation
│   └── lvis_dataset.py              # LVIS dataset for open-vocabulary detection evaluation
│
├── download_data.py                  # Data download script (COCO, LVIS, coco_lvis)
│                                      # Purpose: Download and unpack datasets
│                                      # Execution: python download_data.py --dataset coco_lvis
│
├── train_feature_alignment.py        # Feature alignment pretraining script
│                                      # Purpose: Train KDEOV model (image–text alignment)
│                                      # Execution: python train_feature_alignment.py
│
├── train_detection_finetune.py       # Detection finetuning script (COCO 2017)
│                                      # Purpose: Finetune KDEOV for open-vocabulary detection
│                                      # Execution: python train_detection_finetune.py (see TRAINING_GUIDE.md)
│
├── eval_detection.py                 # Evaluation script (Phase 3)
│                                      # Purpose: mAP / AP@50 on LVIS val or COCO val
│                                      # Execution: python eval_detection.py --checkpoint <path> --dataset lvis
│
├── debug_eval_iou.py                 # Debug script for detection evaluation (IoU / matching)
│                                      # Purpose: Diagnose evaluation pipeline and box matching
│                                      # Execution: python debug_eval_iou.py
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
├── .gitignore                        # Git ignore rules (data dirs, checkpoints, IDE, etc.)
│                                      # Purpose: Exclude generated and local files from version control
│
├── README.md                         # Main project documentation (this file)
│                                      # Purpose: Project overview, installation guide, usage instructions
│
├── TRAINING_GUIDE.md                 # COCO + LVIS training workflow and commands
│                                      # Purpose: End-to-end training, finetuning, and evaluation guide
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
| `models/README.md` | Documentation | Model architecture and API | Reference reading |
| `data/__init__.py` | Python Module | Export dataset APIs | `from data import ...` |
| `data/coco_dataset.py` | Python Module | COCO 2017 dataset | Used via training/eval scripts |
| `data/detection_dataset.py` | Python Module | Detection dataset and collation | Used via training/eval scripts |
| `data/lvis_dataset.py` | Python Module | LVIS dataset for OVOD evaluation | Used via eval scripts |
| `download_data.py` | Executable Script | Download COCO, LVIS, coco_lvis | `python download_data.py --dataset coco_lvis` |
| `train_feature_alignment.py` | Executable Script | Feature alignment pretraining | `python train_feature_alignment.py` |
| `train_detection_finetune.py` | Executable Script | Detection finetuning (COCO 2017) | `python train_detection_finetune.py` (see TRAINING_GUIDE.md) |
| `eval_detection.py` | Executable Script | Evaluation (mAP, AP@50 on val) | `python eval_detection.py --checkpoint <path> --dataset lvis` |
| `debug_eval_iou.py` | Executable Script | Debug evaluation / IoU matching | `python debug_eval_iou.py` |
| `example_usage.py` | Executable Script | Usage examples | `python example_usage.py` |
| `test_scripts/test_environment.py` | Executable Script | Verify CUDA, CLIP, ultralytics | `python test_scripts/test_environment.py` |
| `test_scripts/test_backbone.py` | Executable Script | Test YOLO backbone functionality | `python test_scripts/test_backbone.py` |
| `test_scripts/model_summary.py` | Executable Script | Model architecture analysis and statistics | `python test_scripts/model_summary.py` |
| `requirements.txt` | Configuration File | Dependency management | `pip install -r requirements.txt` |
| `.gitignore` | Configuration File | Git ignore rules | Applied by Git |
| `README.md` | Documentation | Project overview and usage | Reference reading |
| `TRAINING_GUIDE.md` | Documentation | COCO + LVIS training workflow | Reference reading |
| `Development_Log.md` | Documentation | Development records | Reference reading |