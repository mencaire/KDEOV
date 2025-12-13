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

### Documentation
- [FYP Research Proposal](./Documents/FYP%20Research%20Proposal.pdf)
- [FYP Term1 Midterm Report](./Documents/FYP%20Term1%20Midterm%20Report.pdf)
- [FYP Term1 End Report](./Documents/FYP%20Term1%20End%20Report.pdf)

## Next Steps
- Implementation of knowledge distillation framework
- Experimental design and dataset preparation
- Model training and evaluation
- Performance analysis and optimization

## Repository Structure
```
KDEOV/
├── Documents/          # Project documentation and reports
├── README.md          # Project overview and documentation
└── ...
```