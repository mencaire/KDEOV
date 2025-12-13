# Knowledge Distillation for Efficient Open-Vocabulary Vision (KDEOV)
## Transferring CLIP's Semantic Alignment to Lightweight Models

## Group(CY2502) Members
- **PENG, MINQI** (1155191548)
- **Zhu, KEYU** (1155191834)

## Project Overview

This Final Year Project (FYP) focuses on knowledge distillation techniques for efficient open-vocabulary vision systems. The main objective is to transfer CLIP's semantic alignment capabilities to lightweight models, enabling efficient open-vocabulary visual recognition while maintaining high performance.

### Key Research Areas
- Knowledge distillation from large-scale vision-language models (CLIP)
- Efficient open-vocabulary vision systems
- Lightweight model optimization
- Semantic alignment transfer

## Installation

### Environment Setup

Follow these steps to set up the development environment:

1. **Create a conda virtual environment with Python 3.9:**
   ```bash
   conda create -n kdeov python=3.9
   conda activate kdeov
   ```

2. **Install PyTorch:**
   
   For CUDA (GPU support):
   ```bash
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   ```
   
   For CPU only:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```
   
   Or using conda:
   ```bash
   conda install pytorch torchvision -c pytorch
   ```

3. **Install other dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install CLIP:**
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

### Term 2 (In Progress) - Implementation Phase

**Model Implementation (Completed):**
- ✅ **Core Model Components**: Implemented all key components of the KDEOV architecture
  - Frozen CLIP Text Encoder (`FrozenCLIPTextEncoder`)
  - Lightweight Visual Backbone (`LightweightVisualBackbone`) with YOLOv8n/YOLOv5s support
  - Projection Network (`ProjectionNetwork`) for feature alignment
  - Cross-Modal Fusion Module (`CrossModalFusionModule`) with FiLM and Cross-Attention support
- ✅ **Loss Functions**: Implemented comprehensive loss functions
  - Distillation Loss (Cosine and L2 variants)
  - Cross-Modal Alignment Loss (InfoNCE-based)
  - Feature Alignment Loss (combined loss for end-to-end training)
- ✅ **Main Model**: Complete `KDEOVModel` class with full training and inference interface
- ✅ **Training Script**: Feature alignment pretraining script (`train_feature_alignment.py`)
- ✅ **Usage Examples**: Example scripts demonstrating zero-shot classification, text-image retrieval, and forward pass

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
- [FYP Research Proposal](./Documents/FYP%20Research%20Proposal.pdf)
- [FYP Term1 Midterm Report](./Documents/FYP%20Term1%20Midterm%20Report.pdf)
- [FYP Term1 End Report](./Documents/FYP%20Term1%20End%20Report.pdf)
- [Development Log](./Development_Log.md) - Detailed implementation progress and work log
- [Model Architecture Documentation](./models/README.md) - Detailed model component documentation

## Usage

### Quick Start

1. **Initialize the model:**
   ```python
   from models import KDEOVModel
   
   model = KDEOVModel(
       clip_model_name="ViT-B/32",
       backbone_type="yolov8n",
       fusion_type="film",
       device="cuda"
   )
   ```

2. **Zero-shot classification:**
   ```python
   image = ...  # Your image tensor
   class_names = ["cat", "dog", "bird"]
   logits = model.zero_shot_classify(image, class_names)
   ```

3. **Training:**
   ```python
   from train_feature_alignment import train_feature_alignment
   
   train_feature_alignment(
       model=model,
       dataloader=your_dataloader,
       num_epochs=10,
       learning_rate=1e-4,
       device="cuda"
   )
   ```

For more examples, see [`example_usage.py`](./example_usage.py).

## Repository Structure
```
KDEOV/
├── models/                    # Model implementation
│   ├── __init__.py           # Package exports
│   ├── components.py         # Model components (encoders, backbones, fusion)
│   ├── kdeov_model.py       # Main KDEOV model class
│   ├── losses.py             # Loss functions
│   └── README.md             # Model architecture documentation
├── train_feature_alignment.py # Training script
├── example_usage.py          # Usage examples
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── Development_Log.md        # Development log and work progress
├── Documents/                # Project documentation and reports
│   ├── FYP Research Proposal.pdf
│   ├── FYP Term1 Midterm Report.pdf
│   └── FYP Term1 End Report.pdf
└── .vscode/                  # IDE configuration
    └── settings.json         # Python interpreter settings
```