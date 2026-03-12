# KDEOV Training Guide: Open-Vocabulary Object Detection (COCO + LVIS)

This guide provides a step-by-step experimental plan for training KDEOV model using the **COCO + LVIS** strategy—the standard approach for Open-Vocabulary Object Detection (OVOD).

## Table of Contents

1. [Understanding train2017 vs val2017](#1-understanding-train2017-vs-val2017)
2. [COCO + LVIS Strategy Overview](#2-coco--lvis-strategy-overview)
3. [Prerequisites](#3-prerequisites)
4. [Phase 1: Quick Sanity Check (COCO128)](#4-phase-1-quick-sanity-check-coco128)
5. [Phase 2: Full Training (COCO + LVIS)](#5-phase-2-full-training-coco--lvis)
6. [Phase 3: Evaluation (Testing)](#6-phase-3-evaluation-testing)
7. [Troubleshooting](#7-troubleshooting)
8. [Ablation studies (for your FYP thesis)](#8-ablation-studies-for-your-fyp-thesis)

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

**Minimal workflow (two steps):**
1. **Phase 1:** COCO128 — verify pipeline works
2. **Phase 2:** Download `coco_lvis` data, then train with `--dataset coco_lvis`
3. **Phase 3:** Evaluate on LVIS val (OVOD benchmark)

You do **not** need to train on COCO2017 or LVIS separately first. Training on `coco_lvis` once uses both COCO and LVIS annotations in a single run. The options to train on `coco2017` only or `lvis` only are **alternatives** (e.g. for ablation or when you want a baseline with 80 classes only).

---

## 3. Prerequisites (run once)

**Type in terminal (one-time setup):**

```bash
conda activate KDEOV
```
```bash
pip install -r requirements.txt
pip install pycocotools
```
```bash
python test_scripts/test_environment.py
```

### 3.4 Directory Structure

- `models/` — model code
- `datasets/` — data (created by `download_data.py`)
- `weights/` — pretrained weights (CLIP, YOLO; downloaded automatically)

---

## 4. Phase 1: Quick Sanity Check (COCO128)

**Goal:** Verify the training pipeline works before spending hours on full data.

**Type in terminal:**

```bash
python download_data.py --dataset coco128
```
```bash
python train_feature_alignment.py --dataset coco128 --epochs 5 --batch-size 16
```

**Expected:** Training starts, loss decreases, checkpoints saved to `checkpoints/`.

---

## 5. Phase 2: Full Training (COCO + LVIS)

### Step 5.1: Download Data

**Type in terminal (download can take 30–60 minutes):**

```bash
python download_data.py --dataset coco_lvis
```

This creates `datasets/coco2017/` (images + COCO annotations) and `datasets/lvis/annotations/` (LVIS JSONs).

### Step 5.2: Train on COCO + LVIS

**Type in terminal:**

```bash
python train_feature_alignment.py --dataset coco_lvis --split train --epochs 10 --batch-size 32 --save-path checkpoints/kdeov_coco_lvis
```

This uses both COCO and LVIS annotations in a single training run. **This is all you need** for OVOD after the Phase 1 sanity check.

### Optional: Step-by-Step Download

If you prefer to download in stages:

```bash
# 1. COCO2017 (images + annotations)
python download_data.py --dataset coco2017 --parts train2017 val2017 annotations

# 2. LVIS annotations (requires COCO2017 images)
python download_data.py --dataset lvis
```

### Optional: Other Training Modes

- **COCO only (80 classes):** `python train_feature_alignment.py --dataset coco2017 --split train --epochs 10 --save-path checkpoints/kdeov_coco`
- **LVIS only (1,203 classes):** `python train_feature_alignment.py --dataset lvis --split train --epochs 10 --save-path checkpoints/kdeov_lvis`

Use these only if you need a baseline or an ablation; the minimal path is coco128 → coco_lvis.

---

## 6. Phase 3: Evaluation (Testing)

**Goal:** Run your trained model on the validation set and report evaluation metrics for your report/paper.

### 6.1 What to Evaluate On (Which Data)

| Where to evaluate | Data | When to use |
|-------------------|------|--------------|
| **LVIS val** | val2017 images + `lvis_v1_val.json` (1,203 classes) | **Primary.** Use this for OVOD results. Same images as COCO val, but LVIS annotations. |
| **COCO val** | val2017 images + `instances_val2017.json` (80 classes) | **Optional.** For comparison with standard COCO 80-class baselines. |

Always use the **validation set** (val2017). Never evaluate on train2017 (that would be data leakage).

### 6.2 Evaluation Metrics to Report

Report at least:

| Metric | Meaning |
|--------|---------|
| **mAP** | Mean Average Precision (IoU 0.5:0.95). Main detection metric. |
| **AP@50** | AP at IoU threshold 0.5. Commonly reported. |

If you evaluate on **LVIS val**, also report (when possible):

| Metric | Meaning |
|--------|---------|
| **AP** | Overall AP on LVIS. |
| **AP_rare** | AP on rare categories (long-tail). Important for OVOD. |
| **AP_common** | AP on common categories. |
| **AP_frequent** | AP on frequent categories. |

These are the numbers you put in your “Experimental Results” section.

### 6.3 Run the Evaluation Script

**Type in terminal (LVIS val — primary):**

```bash
python eval_detection.py --checkpoint checkpoints/kdeov_coco_lvis_epoch_10.pt --dataset lvis
```

**Type in terminal (COCO val — optional):**

```bash
python eval_detection.py --checkpoint checkpoints/kdeov_coco_lvis_epoch_10.pt --dataset coco
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | (required) | Path to your trained `.pt` checkpoint |
| `--data-root` | `datasets` | Root directory for datasets |
| `--dataset` | `lvis` | `lvis` or `coco` |
| `--batch-size` | 32 | Batch size for inference |
| `--score-thresh` | 0.1 | Minimum detection score |
| `--backbone` | yolov8n | Must match the backbone used when training |

**Requirements:** Ensure you have run `python download_data.py --dataset coco_lvis` so that `datasets/coco2017/val2017/` and `datasets/lvis/annotations/lvis_v1_val.json` (and for COCO, `datasets/coco2017/annotations/instances_val2017.json`) exist.

- **COCO:** The script uses `pycocotools` to compute **mAP** and **AP@50** and prints them.
- **LVIS:** If the `lvis` package is installed (`pip install lvis`), the script runs **LVISEval** and prints AP (and AP_rare/common/frequent when available). If not, it saves a sample of predictions to `eval_lvis_results.json` and reminds you to install `lvis` for full metrics.

### 6.6 What Counts as “Experimental Results”

For your report/paper, the **experimental results** typically include:

1. **Checkpoints** — e.g. `kdeov_coco_lvis_epoch_10.pt` (trained model).
2. **Training curves** — e.g. `checkpoints/kdeov_training_curves.png` (loss vs epoch).
3. **Evaluation metrics** — table of **mAP**, **AP@50**, and (on LVIS) **AP / AP_rare / AP_common / AP_frequent** on **LVIS val** (and optionally on **COCO val**).

The metrics table is what reviewers and readers will look at as the main “results”; checkpoints and curves support reproducibility and analysis.

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

### LVIS metrics not printed (only COCO works)

For full LVIS metrics (AP, AP_rare, etc.), install the LVIS API: `pip install lvis`. Without it, `eval_detection.py --dataset lvis` still runs detection but only saves a sample JSON and does not compute LVIS AP.

### Loss not decreasing

- Try lower learning rate: `--lr 5e-5`
- Train for more epochs
- Check data loading (no empty batches)

---

## Quick Reference: Command Summary

**Minimal path (coco128 → coco_lvis → evaluation):**

| Step | Task | Command / Action |
|------|------|------------------|
| 1 | Download COCO128 | `python download_data.py --dataset coco128` |
| 1 | Train sanity check | `python train_feature_alignment.py --dataset coco128 --epochs 5` |
| 2 | Download COCO + LVIS | `python download_data.py --dataset coco_lvis` |
| 2 | Full OVOD training | `python train_feature_alignment.py --dataset coco_lvis --split train --epochs 10 --save-path checkpoints/kdeov_coco_lvis` |
| 3 | Evaluation (testing) | `python eval_detection.py --checkpoint checkpoints/kdeov_coco_lvis_epoch_10.pt --dataset lvis` (and optionally `--dataset coco`). See Phase 3 (§6). |

**Other options:**

| Task | Command |
|------|---------|
| Download COCO2017 only | `python download_data.py --dataset coco2017 --parts train2017 val2017 annotations` |
| Download LVIS only | `python download_data.py --dataset lvis` |
| Train on COCO2017 only | `python train_feature_alignment.py --dataset coco2017 --split train --epochs 10` |
| Train on LVIS only | `python train_feature_alignment.py --dataset lvis --split train --epochs 10` |
| Train with dummy data | `python train_feature_alignment.py --dataset dummy --epochs 3` |

---

## Recommended Experiment Order — Copy-paste commands

Run these in order. Copy each line into your terminal (from project root, with `conda activate KDEOV` already done).

**Step 1 — Sanity check (COCO128):**
```bash
python download_data.py --dataset coco128
```
```bash
python train_feature_alignment.py --dataset coco128 --epochs 5 --batch-size 16
```

**Step 2 — Full data and training (COCO + LVIS):**
```bash
python download_data.py --dataset coco_lvis
```
```bash
python train_feature_alignment.py --dataset coco_lvis --split train --epochs 10 --batch-size 32 --save-path checkpoints/kdeov_coco_lvis
```

**Step 3 — Evaluation (get mAP, AP@50):**
```bash
python eval_detection.py --checkpoint checkpoints/kdeov_coco_lvis_epoch_10.pt --dataset lvis
```
*(Optional: also run with `--dataset coco` for 80-class metrics.)*
```bash
python eval_detection.py --checkpoint checkpoints/kdeov_coco_lvis_epoch_10.pt --dataset coco
```

No other scripts are required. The numbers printed at the end of the eval commands are what you report as your experimental results.

---

## 8. Ablation studies

**What is an ablation?** An ablation is when you change one thing (e.g. backbone or fusion type), keep everything else the same, and compare metrics to see how much that choice matters. You do **not** need any new scripts: use the same `train_feature_alignment.py` and `eval_detection.py` with different flags (e.g. `--backbone yolov5s` or `--fusion cross_attention`).

Typical FYP/paper ablations show how each design choice affects the final metrics. The codebase **already supports** several ablations via existing options; a couple of extra CLI flags give you more without big code changes.

### 8.1 What you can vary **without code changes**

| Variable | How to change | Purpose |
|----------|----------------|---------|
| **Backbone** | `--backbone yolov8n` vs `--backbone yolov5s` | Effect of visual backbone size (lighter vs heavier). |
| **Training data** | `--dataset coco2017` vs `--dataset lvis` vs `--dataset coco_lvis` | Effect of 80-class vs 1203-class vs combined. |
| **Epochs** | `--epochs 5` vs `--epochs 10` vs `--epochs 20` | Effect of training length. |
| **Learning rate** | `--lr 1e-4` vs `--lr 5e-5` | Sensitivity to learning rate. |
| **Batch size** | `--batch-size 16` vs `--batch-size 32` | Can affect convergence (and GPU memory). |
| **Fusion type** | `--fusion film` vs `--fusion cross_attention` | FiLM vs cross-attention (model supports both). |

Use different **`--save-path`** per run so you keep separate checkpoints. Example commands (run after Step 2 data is ready):

- **Backbone ablation (yolov5s):**  
  `python train_feature_alignment.py --dataset coco_lvis --split train --epochs 10 --batch-size 32 --backbone yolov5s --save-path checkpoints/ablation_yolov5s`  
  then: `python eval_detection.py --checkpoint checkpoints/ablation_yolov5s_epoch_10.pt --dataset lvis --backbone yolov5s`

- **Fusion ablation (cross_attention):**  
  `python train_feature_alignment.py --dataset coco_lvis --split train --epochs 10 --batch-size 32 --fusion cross_attention --save-path checkpoints/ablation_cross_attn`  
  then: `python eval_detection.py --checkpoint checkpoints/ablation_cross_attn_epoch_10.pt --dataset lvis --fusion cross_attention`

### 8.2 Optional: expose more variables for ablations

| Variable | Where | Purpose |
|----------|--------|---------|
| **Loss weights** | In `train_feature_alignment.py`, `FeatureAlignmentLoss(distillation_weight=..., alignment_weight=...)` is fixed. Expose them as `--distillation-weight` and `--alignment-weight` if you want to ablate their ratio. | Ablation: importance of distillation vs alignment. |

So the setup **leaves room for ablations**: backbone, data, epochs, lr, batch size, and fusion are already configurable from the CLI; only loss weights need a small code change if you want that in the thesis.

### 8.3 Suggested table for the thesis

- **Main results:** One table with mAP / AP@50 (and LVIS AP, AP_rare, etc.) for your best run (e.g. coco_lvis, yolov8n, 10 epochs).
- **Ablation tables:**
  - Backbone: rows for `yolov8n` vs `yolov5s` (same data, same epochs).
  - Data: rows for `coco2017` vs `lvis` vs `coco_lvis` (same backbone, same epochs).
  - Fusion: rows for `film` vs `cross_attention` (use `--fusion film` / `--fusion cross_attention`).
  - Optionally: learning rate or epochs (e.g. 5 vs 10 vs 20).

Always report the **same evaluation** (e.g. LVIS val mAP / AP@50) so numbers are comparable across rows.
