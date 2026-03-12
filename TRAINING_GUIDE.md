# KDEOV Training Guide: Open-Vocabulary Object Detection (COCO + LVIS)

This guide provides a step-by-step experimental plan for training KDEOV model using the **COCO + LVIS** strategy—the standard approach for Open-Vocabulary Object Detection (OVOD).

---

## Table of Contents

1. [Understanding train2017 vs val2017](#1-understanding-train2017-vs-val2017)
2. [COCO + LVIS Strategy Overview](#2-coco--lvis-strategy-overview)
3. [Prerequisites](#3-prerequisites)
4. [Phase 1: Quick Sanity Check (COCO128)](#4-phase-1-quick-sanity-check-coco128)
5. [Phase 2: Full Training (COCO + LVIS)](#5-phase-2-full-training-coco--lvis)
6. [Phase 3: Evaluation](#6-phase-3-evaluation)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Understanding train2017 vs val2017

| Split | When to Use | Purpose |
|-------|-------------|---------|
| **train2017** | During training | Update model weights. The model learns from these images. |
| **val2017** | After training / during evaluation | Evaluate performance. **Never** use for training—it would cause "data leakage" and overestimate your model's real-world performance. |

**Rule of thumb:**
- **Training** → always use `train2017`
- **Validation / Testing** → always use `val2017`

COCO128 only has a train split (128 images). COCO2017 and LVIS both use train2017/val2017.

---

## 2. COCO + LVIS Strategy Overview

| Dataset | Classes | Images | Role |
|---------|---------|--------|------|
| **COCO** | 80 | ~118k train, ~5k val | Baseline training, debugging |
| **LVIS** | 1,203 | ~100k train, ~20k val | OVOD benchmark, large vocabulary |
| **COCO + LVIS** | Combined | Same COCO images | Train with both annotations for best OVOD performance |

**LVIS uses COCO 2017 images**—you only need to download LVIS annotations separately. The images are shared.

**Recommended workflow:**
1. **Phase 1:** COCO128 — verify pipeline
2. **Phase 2a:** COCO2017 — baseline training
3. **Phase 2b:** Download LVIS, train on LVIS or coco_lvis
4. **Phase 3:** Evaluate on LVIS val (OVOD benchmark)

---

## 3. Prerequisites

### 3.1 Environment

```bash
conda activate KDEOV
```

### 3.2 Verify Installation

```bash
python test_scripts/test_environment.py
```

### 3.3 Install Dependencies

```bash
pip install -r requirements.txt
pip install pycocotools   # For COCO/LVIS annotations
```

### 3.4 Directory Structure

- `models/` — model code
- `datasets/` — data (created by `download_data.py`)
- `weights/` — pretrained weights (CLIP, YOLO; downloaded automatically)

---

## 4. Phase 1: Quick Sanity Check (COCO128)

**Goal:** Verify the training pipeline works before spending hours on full data.

### Step 4.1: Download COCO128

```bash
python download_data.py --dataset coco128
```

### Step 4.2: Run Training (5 epochs)

```bash
python train_feature_alignment.py --dataset coco128 --epochs 5 --batch-size 16
```

**Expected:** Training starts, loss decreases, checkpoints saved to `checkpoints/`.

---

## 5. Phase 2: Full Training (COCO + LVIS)

### Option A: One-Command Download (Recommended)

Download COCO2017 images + COCO annotations + LVIS annotations in one go:

```bash
python download_data.py --dataset coco_lvis
```

This creates:
- `datasets/coco2017/train2017/`, `val2017/` — images (~19 GB)
- `datasets/coco2017/annotations/` — COCO annotations
- `datasets/lvis/annotations/` — LVIS annotations (lvis_v1_train.json, lvis_v1_val.json)

**Note:** Download can take 30–60 minutes.

### Option B: Step-by-Step Download

```bash
# 1. COCO2017 (images + annotations)
python download_data.py --dataset coco2017 --parts train2017 val2017 annotations

# 2. LVIS annotations (requires COCO2017 images)
python download_data.py --dataset lvis
```

### Step 5.1: Train on COCO2017 (Baseline)

```bash
python train_feature_alignment.py --dataset coco2017 --split train --epochs 10 --batch-size 32 --save-path checkpoints/kdeov_coco
```

### Step 5.2: Train on LVIS (OVOD Benchmark)

```bash
python train_feature_alignment.py --dataset lvis --split train --epochs 10 --batch-size 32 --save-path checkpoints/kdeov_lvis
```

### Step 5.3: Train on COCO + LVIS (Combined, Best for OVOD)

```bash
python train_feature_alignment.py --dataset coco_lvis --split train --epochs 10 --batch-size 32 --save-path checkpoints/kdeov_coco_lvis
```

This uses both COCO and LVIS annotations (same images, different class vocabularies) for richer training.

---

## 6. Phase 3: Evaluation

**Goal:** Evaluate your trained model on validation data.

### Step 6.1: Load Trained Checkpoint

Use the checkpoint from `checkpoints/` (e.g. `kdeov_lvis_epoch_10.pt`).

### Step 6.2: Evaluate on LVIS Val (OVOD Benchmark)

Use **val2017** images with LVIS class names for open-vocabulary evaluation:

```python
from models import KDEOVModel
import torch

model = KDEOVModel(clip_model_name="ViT-B/32", backbone_type="yolov8n", fusion_type="film")
model.load_state_dict(torch.load("checkpoints/kdeov_lvis_epoch_10.pt")["model_state_dict"])

# Use val2017 images + LVIS class names for OVOD evaluation
boxes, scores, labels = model.open_vocabulary_detect(
    images,
    class_names=["person", "car", "dog", "backpack", "umbrella", ...]  # LVIS 1203 classes
)
```

**Important:** Always use `val2017` for evaluation—never `train2017`.

---

## 7. Troubleshooting

### "LVIS 需要 COCO 2017 图像"

Run COCO2017 first: `python download_data.py --dataset coco2017 --parts train2017 val2017`

### "LVIS annotations not found"

Run: `python download_data.py --dataset lvis`

### Out of GPU Memory

- Reduce `--batch-size` to 16 or 8
- Use `--backbone yolov8n` (smaller than yolov5s)

### "No module named 'pycocotools'"

Run: `pip install pycocotools`

### Loss not decreasing

- Try lower learning rate: `--lr 5e-5`
- Train for more epochs
- Check data loading (no empty batches)

---

## Quick Reference: Command Summary

| Task | Command |
|------|---------|
| Download COCO128 | `python download_data.py --dataset coco128` |
| Download COCO2017 | `python download_data.py --dataset coco2017 --parts train2017 val2017 annotations` |
| Download LVIS | `python download_data.py --dataset lvis` |
| Download COCO + LVIS (all) | `python download_data.py --dataset coco_lvis` |
| Train on COCO128 (quick test) | `python train_feature_alignment.py --dataset coco128 --epochs 5` |
| Train on COCO2017 | `python train_feature_alignment.py --dataset coco2017 --split train --epochs 10` |
| Train on LVIS | `python train_feature_alignment.py --dataset lvis --split train --epochs 10` |
| Train on COCO + LVIS | `python train_feature_alignment.py --dataset coco_lvis --split train --epochs 10` |
| Train with dummy data | `python train_feature_alignment.py --dataset dummy --epochs 3` |

---

## Recommended Experiment Order

1. **Day 1:** Phase 1 (COCO128) — confirm pipeline works
2. **Day 2:** Run `python download_data.py --dataset coco_lvis`
3. **Day 3:** Phase 2 — train on coco2017, then lvis or coco_lvis
4. **Day 4+:** Phase 3 evaluation on LVIS val, tune hyperparameters

Good luck with your FYP!
