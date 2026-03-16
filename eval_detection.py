"""
Evaluation script for KDEOV open-vocabulary object detection (Phase 3).

Evaluation logic: run the KDEoV model on val images to get predicted bboxes
(boxes, scores, labels); compare these to COCO/LVIS ground-truth bboxes using
IoU; AP/AR are the standard detection metrics (precision/recall over IoU thresholds).
Predictions are scaled from model input size (224x224) to original image size before
comparison so that IoU is computed in the same coordinate system as the GT.

Computes mAP, AP@50, and (for LVIS) AP_rare/common/frequent on val2017.
Usage:
  # LVIS val (1,203 classes) — primary OVOD benchmark
  python eval_detection.py --checkpoint checkpoints/kdeov_coco_lvis_epoch_10.pt --dataset lvis

  # COCO val (80 classes) — optional baseline comparison
  python eval_detection.py --checkpoint checkpoints/kdeov_coco_lvis_epoch_10.pt --dataset coco
"""

import argparse
import json
from pathlib import Path

import numpy as np

# NumPy 1.24+ removed np.float; lvis 0.5.3 still uses it. Restore alias so we don't patch lvis.
if not hasattr(np, "float"):
    np.float = np.float64
if not hasattr(np, "int"):
    np.int = np.int64

import torch
from PIL import Image
from tqdm import tqdm

from models import KDEOVModel

# CLIP normalization
IMAGE_SIZE = 224
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    t = (t - MEAN) / STD
    return t


def load_coco_val(data_root: str):
    """Load COCO val image paths and category info. Returns (img_infos, class_names, name_to_cat_id)."""
    from pycocotools.coco import COCO

    data_root = Path(data_root)
    ann_file = data_root / "coco2017" / "annotations" / "instances_val2017.json"
    img_dir = data_root / "coco2017" / "val2017"
    if not ann_file.exists() or not img_dir.exists():
        raise FileNotFoundError(f"COCO val not found. Run: python download_data.py --dataset coco2017 --parts val2017 annotations")

    coco = COCO(str(ann_file))
    cat_ids = sorted(coco.getCatIds())
    cats = coco.loadCats(cat_ids)
    class_names = [c["name"] for c in cats]
    name_to_cat_id = {c["name"]: c["id"] for c in cats}

    img_ids = coco.getImgIds()
    img_infos = []
    for iid in img_ids:
        info = coco.loadImgs(iid)[0]
        path = img_dir / info["file_name"]
        if path.exists():
            img_infos.append({
                "id": iid, "path": str(path), "file_name": info["file_name"],
                "width": info["width"], "height": info["height"],
            })
    return img_infos, class_names, name_to_cat_id


def load_lvis_val(data_root: str):
    """Load LVIS val image paths and category info. Returns (img_infos, class_names, name_to_cat_id)."""
    data_root = Path(data_root)
    ann_file = data_root / "lvis" / "annotations" / "lvis_v1_val.json"
    img_dir = data_root / "coco2017" / "val2017"
    if not ann_file.exists() or not img_dir.exists():
        raise FileNotFoundError(f"LVIS val not found. Run: python download_data.py --dataset coco_lvis")

    with open(ann_file) as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    categories = {c["id"]: c for c in data["categories"]}
    cat_ids = sorted(categories.keys())
    class_names = []
    name_to_cat_id = {}
    for cid in cat_ids:
        c = categories[cid]
        name = c.get("name") or (c.get("synonyms", ["object"])[0])
        class_names.append(name)
        name_to_cat_id[name] = cid

    img_infos = []
    for iid in images.keys():
        img = images[iid]
        # LVIS uses COCO images; path by COCO convention (id zero-padded to 12 digits)
        path = img_dir / f"{iid:012d}.jpg"
        if path.exists():
            img_infos.append({
                "id": iid, "path": str(path), "file_name": f"{iid:012d}.jpg",
                "width": img["width"], "height": img["height"],
            })
    return img_infos, class_names, name_to_cat_id


def run_eval_coco(model, img_infos, class_names, name_to_cat_id, device, data_root, batch_size=32, score_thresh=0.1):
    """Run detection and evaluate with pycocotools COCOeval."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    data_root = Path(data_root)
    ann_file = data_root / "coco2017" / "annotations" / "instances_val2017.json"
    coco_gt = COCO(str(ann_file))

    results = []
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(img_infos), batch_size), desc="COCO val"):
            batch_infos = img_infos[start : start + batch_size]
            images = torch.stack([load_image(info["path"]) for info in batch_infos]).to(device)
            out_list = model.open_vocabulary_detect(
                images, class_names=class_names, score_threshold=score_thresh, nms_threshold=0.5
            )
            for info, out in zip(batch_infos, out_list):
                boxes = out["boxes"].cpu()
                scores = out["scores"].cpu()
                labels = out["labels"]
                orig_w = info["width"]
                orig_h = info["height"]
                scale_x = orig_w / IMAGE_SIZE
                scale_y = orig_h / IMAGE_SIZE
                for k in range(len(scores)):
                    cat_name = labels[k]
                    cat_id = name_to_cat_id.get(cat_name)
                    if cat_id is None:
                        continue
                    x1, y1, x2, y2 = boxes[k].tolist()
                    # Scale from model input size (224x224) to original image size for COCO eval
                    x1 = x1 * scale_x
                    y1 = y1 * scale_y
                    x2 = x2 * scale_x
                    y2 = y2 * scale_y
                    results.append({
                        "image_id": info["id"],
                        "category_id": cat_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(scores[k]),
                    })
    out_path = "eval_coco_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved {len(results)} predictions to {out_path}.")
    if len(results) == 0:
        print("No detections above score threshold. COCO metrics set to 0. Try: --score-thresh 0.01")
        return (0.0,) * 12
    coco_dt = coco_gt.loadRes(results)
    eval_obj = COCOeval(coco_gt, coco_dt, "bbox")
    eval_obj.evaluate()
    eval_obj.accumulate()
    eval_obj.summarize()
    return eval_obj.stats


def run_eval_lvis(model, img_infos, class_names, name_to_cat_id, device, data_root, batch_size=32, score_thresh=0.1):
    """Run detection and evaluate with LVIS API if available, else save results JSON."""
    data_root = Path(data_root)
    results = []
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(img_infos), batch_size), desc="LVIS val"):
            batch_infos = img_infos[start : start + batch_size]
            images = torch.stack([load_image(info["path"]) for info in batch_infos]).to(device)
            out_list = model.open_vocabulary_detect(
                images, class_names=class_names, score_threshold=score_thresh, nms_threshold=0.5
            )
            for info, out in zip(batch_infos, out_list):
                boxes = out["boxes"].cpu()
                scores = out["scores"].cpu()
                labels = out["labels"]
                orig_w = info["width"]
                orig_h = info["height"]
                scale_x = orig_w / IMAGE_SIZE
                scale_y = orig_h / IMAGE_SIZE
                for k in range(len(scores)):
                    cat_name = labels[k]
                    cat_id = name_to_cat_id.get(cat_name)
                    if cat_id is None:
                        continue
                    x1, y1, x2, y2 = boxes[k].tolist()
                    # Scale from model input size (224x224) to original image size for LVIS eval
                    x1 = x1 * scale_x
                    y1 = y1 * scale_y
                    x2 = x2 * scale_x
                    y2 = y2 * scale_y
                    results.append({
                        "image_id": info["id"],
                        "category_id": cat_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(scores[k]),
                    })

    out_path = "eval_lvis_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    n = len(results)
    if n == 0:
        print(f"No detections above score threshold ({score_thresh}). Saved empty list to {out_path}.")
        print("Try a lower threshold, e.g.: --score-thresh 0.01")
    else:
        print(f"Saved {n} predictions to {out_path}.")

    try:
        from lvis import LVIS
        from lvis.eval import LVISEval

        ann_file = data_root / "lvis" / "annotations" / "lvis_v1_val.json"
        lvis_gt = LVIS(str(ann_file))
        # LVISEval accepts lvis_dt as list of dicts (or path); no loadRes in pip lvis
        lvis_eval = LVISEval(lvis_gt, results, "bbox")
        lvis_eval.run()
        lvis_eval.print_results()
        res = getattr(lvis_eval, "results", None)
        if n > 10000:
            print("\nTip: If AP is 0, many low-score detections can swamp true positives. Try: --score-thresh 0.2 or 0.25")
        return res
    except ImportError:
        print("LVIS API not installed. Install with: pip install lvis")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate KDEOV on COCO or LVIS val")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--data-root", type=str, default="datasets", help="Root directory for datasets")
    parser.add_argument("--dataset", type=str, choices=["coco", "lvis"], default="lvis", help="Evaluate on COCO val (80 cls) or LVIS val (1203 cls)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--score-thresh", type=float, default=0.01,
                        help="Min detection score. If AP is 0, try 0.2–0.25 to reduce false positives.")
    parser.add_argument("--backbone", type=str, default="yolov8n", choices=["yolov8n", "yolov5s"])
    parser.add_argument("--fusion", type=str, default="film", choices=["film", "cross_attention"], help="Must match training")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = KDEOVModel(clip_model_name="ViT-B/32", backbone_type=args.backbone, fusion_type=args.fusion, weights_dir="weights")
    load_ret = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if load_ret.missing_keys:
        print(f"Note: loaded with strict=False (missing: bbox_regression_head etc.). Refined boxes will not be used.")
    model = model.to(device)
    model.eval()

    if args.dataset == "coco":
        img_infos, class_names, name_to_cat_id = load_coco_val(args.data_root)
        print(f"COCO val: {len(img_infos)} images, {len(class_names)} classes")
        stats = run_eval_coco(model, img_infos, class_names, name_to_cat_id, device, args.data_root, args.batch_size, args.score_thresh)
        if stats is not None:
            print("\n--- COCO metrics ---")
            print(f"  mAP (0.5:0.95): {stats[0]:.4f}")
            print(f"  AP@50:          {stats[1]:.4f}")
            print(f"  AP@75:          {stats[2]:.4f}")
    else:
        img_infos, class_names, name_to_cat_id = load_lvis_val(args.data_root)
        print(f"LVIS val: {len(img_infos)} images, {len(class_names)} classes")
        run_eval_lvis(model, img_infos, class_names, name_to_cat_id, device, args.data_root, args.batch_size, args.score_thresh)


if __name__ == "__main__":
    main()
