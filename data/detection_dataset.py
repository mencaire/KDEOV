"""
Detection dataset for fine-tuning: loads images + bounding boxes + category IDs.

Uses COCO train2017 (instances_train2017.json). No extra download if you already
ran: python download_data.py --dataset coco_lvis
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .coco_dataset import COCO_CLASS_NAMES

IMAGE_SIZE = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class COCODetectionDataset(Dataset):
    """
    COCO train with bbox annotations for detection fine-tuning.
    Returns (image_tensor, boxes_xyxy, category_indices).
    - image_tensor: [3, 224, 224], normalized for CLIP
    - boxes_xyxy: [N, 4] in image coords (224x224)
    - category_indices: [N] int in 0..79 (COCO 80 classes)
    """

    def __init__(self, data_root: str = "datasets", max_boxes: int = 50):
        self.data_root = Path(data_root)
        self.max_boxes = max_boxes
        img_dir = self.data_root / "coco2017" / "train2017"
        ann_file = self.data_root / "coco2017" / "annotations" / "instances_train2017.json"
        if not ann_file.exists() or not img_dir.exists():
            raise FileNotFoundError(
                "COCO train not found. Run: python download_data.py --dataset coco_lvis"
            )
        from pycocotools.coco import COCO
        self.coco = COCO(str(ann_file))
        self.img_ids = self.coco.getImgIds()
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_idx = {cid: i for i, cid in enumerate(self.cat_ids)}
        self.class_names = COCO_CLASS_NAMES
        self.img_dir = img_dir

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_id = self.img_ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_dir / info["file_name"]
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        img_arr = np.array(img).astype(np.float32) / 255.0
        img_arr = (img_arr - MEAN) / STD
        image_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float()

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        scale_x = IMAGE_SIZE / max(orig_w, 1)
        scale_y = IMAGE_SIZE / max(orig_h, 1)
        for ann in anns:
            if "bbox" not in ann:
                continue
            x, y, w, h = ann["bbox"]
            cat_id = ann.get("category_id")
            if cat_id not in self.cat_id_to_idx:
                continue
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_id_to_idx[cat_id])
        if not boxes:
            boxes = [[0.0, 0.0, 1.0, 1.0]]
            labels = [0]
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        if len(boxes) > self.max_boxes:
            boxes = boxes[: self.max_boxes]
            labels = labels[: self.max_boxes]
        return image_tensor, boxes, labels


def collate_detection(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Pad boxes and labels to max N in batch."""
    images = torch.stack([b[0] for b in batch])
    max_n = max(b[1].size(0) for b in batch)
    boxes_list = []
    labels_list = []
    for _, boxes, labels in batch:
        n = boxes.size(0)
        if n < max_n:
            pad_boxes = torch.zeros(max_n - n, 4, dtype=boxes.dtype)
            pad_labels = torch.full((max_n - n,), -1, dtype=torch.long)
            boxes = torch.cat([boxes, pad_boxes], dim=0)
            labels = torch.cat([labels, pad_labels], dim=0)
        boxes_list.append(boxes)
        labels_list.append(labels)
    return images, torch.stack(boxes_list), torch.stack(labels_list)
