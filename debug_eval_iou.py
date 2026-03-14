"""
Debug script: check if any prediction has IoU >= 0.5 with a GT of the same category.
Helps determine why AP is 0 (coords wrong vs grid boxes never overlap enough).
Run from project root: python debug_eval_iou.py
"""

import json
from pathlib import Path


def bbox_xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def iou_xyxy(box1, box2):
    """Compute IoU of two boxes in xyxy format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def main():
    data_root = Path("datasets")
    ann_file = data_root / "lvis" / "annotations" / "lvis_v1_val.json"
    results_path = Path("eval_lvis_results.json")

    if not ann_file.exists():
        print(f"LVIS annotations not found: {ann_file}")
        return
    if not results_path.exists():
        print(f"Results not found. Run eval first: {results_path}")
        return

    with open(ann_file) as f:
        data = json.load(f)

    # Build list of GT annotations per image: image_id -> [(category_id, bbox_xyxy), ...]
    gt_by_img = {}
    for ann in data["annotations"]:
        iid = ann["image_id"]
        if iid not in gt_by_img:
            gt_by_img[iid] = []
        gt_by_img[iid].append((ann["category_id"], bbox_xywh_to_xyxy(ann["bbox"])))

    # Load our predictions (full file; can be large)
    print("Loading predictions...")
    with open(results_path) as f:
        preds = json.load(f)
    print(f"Loaded {len(preds)} predictions")

    # Group predictions by image_id
    pred_by_img = {}
    for p in preds:
        iid = p["image_id"]
        if iid not in pred_by_img:
            pred_by_img[iid] = []
        pred_by_img[iid].append({
            "category_id": p["category_id"],
            "bbox_xyxy": bbox_xywh_to_xyxy(p["bbox"]),
            "score": p["score"],
        })

    # For each image that has both GT and preds, compute max same-category IoU
    iou_threshold = 0.5
    total_matches = 0
    total_gt = 0
    max_iou_seen = 0.0
    num_images_with_gt = 0
    sample_printed = 0

    for iid, gt_list in gt_by_img.items():
        if iid not in pred_by_img:
            continue
        num_images_with_gt += 1
        pred_list = pred_by_img[iid]
        for cat_id, gt_xyxy in gt_list:
            total_gt += 1
            best_iou = 0.0
            for p in pred_list:
                if p["category_id"] != cat_id:
                    continue
                iou = iou_xyxy(p["bbox_xyxy"], gt_xyxy)
                best_iou = max(best_iou, iou)
                max_iou_seen = max(max_iou_seen, iou)
            if best_iou >= iou_threshold:
                total_matches += 1
            if sample_printed < 3 and (best_iou > 0.1 or len(gt_list) <= 2):
                print(f"  image_id={iid} cat_id={cat_id} best_iou={best_iou:.3f} gt_xyxy={[round(x,1) for x in gt_xyxy]}")
                sample_printed += 1

    print("\n--- Debug summary ---")
    print(f"Images with both GT and preds: {num_images_with_gt}")
    print(f"GT instances checked: {total_gt}")
    print(f"GT instances with >= one pred at IoU >= 0.5 (same cat): {total_matches}")
    print(f"Max IoU seen (any pred vs GT, same cat): {max_iou_seen:.4f}")
    if max_iou_seen < 0.5:
        print("\n-> No prediction reaches IoU 0.5 with GT. Likely cause: grid default boxes")
        print("   don't align with object boundaries. Consider: box regression head or")
        print("   larger cell_scale / different grid so some cell overlaps objects enough.")
    else:
        print("\n-> Some predictions do reach IoU >= 0.5; if AP is still 0, check LVIS eval format.")


if __name__ == "__main__":
    main()
