"""
COCO Dataset for KDEOV Feature Alignment Training

Supports:
- COCO128: YOLO format (images/train2017, labels/train2017)
- COCO2017: COCO format (train2017, val2017 + annotations JSON)

Usage:
  train split -> for training (updates model weights)
  val split   -> for validation (evaluate, never train on it)
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import clip  # type: ignore

# COCO 80 class names (index 0-79, matches YOLO/COCO class_id)
COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# CLIP expects 224x224
IMAGE_SIZE = 224


class COCODataset(Dataset):
    """
    COCO dataset for image-text pair training.
    
    For each image, creates text from object class names in that image.
    E.g., "a photo of a person, a car, a dog"
    
    Supports:
    - split="train": use train2017 (for training)
    - split="val": use val2017 (for validation only, do NOT train on this)
    """

    def __init__(
        self,
        data_root: str,
        dataset_name: str = "coco128",
        split: str = "train",
        max_text_length: int = 77,
        tokenizer=None,
    ):
        """
        Args:
            data_root: Root directory (e.g. "datasets")
            dataset_name: "coco128" or "coco2017"
            split: "train" or "val"
            max_text_length: CLIP context length (77)
            tokenizer: CLIP tokenizer (clip.tokenize)
        """
        self.data_root = Path(data_root)
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer = tokenizer or clip.tokenize
        self.max_text_length = max_text_length

        if dataset_name == "coco128":
            self._setup_coco128()
        elif dataset_name == "coco2017":
            self._setup_coco2017()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def _setup_coco128(self) -> None:
        """Setup COCO128 (YOLO format). COCO128 only has train split."""
        base = self.data_root / "coco128"
        img_dir = base / "images" / "train2017"
        label_dir = base / "labels" / "train2017"

        if not img_dir.exists():
            raise FileNotFoundError(
                f"COCO128 not found at {img_dir}. Run: python download_data.py --dataset coco128"
            )

        self.image_paths: List[str] = []
        self.label_paths: List[Optional[str]] = []

        for img_path in sorted(img_dir.glob("*.jpg")):
            self.image_paths.append(str(img_path))
            label_path = label_dir / (img_path.stem + ".txt")
            self.label_paths.append(str(label_path) if label_path.exists() else None)

        self.use_yolo_labels = True
        self.coco = None
        self.img_ids = []
        self.use_coco_api = False

    def _setup_coco2017(self) -> None:
        """Setup COCO2017 (COCO format with annotations)."""
        base = self.data_root / "coco2017"
        if self.split == "train":
            img_dir = base / "train2017"
            ann_file = base / "annotations" / "instances_train2017.json"
        else:
            img_dir = base / "val2017"
            ann_file = base / "annotations" / "instances_val2017.json"

        if not img_dir.exists():
            raise FileNotFoundError(
                f"COCO2017 {self.split} not found at {img_dir}. "
                f"Run: python download_data.py --dataset coco2017 --parts train2017 val2017 annotations"
            )

        self.image_paths = []
        self.label_paths = []
        self.use_yolo_labels = False

        # Try COCO JSON annotations
        if ann_file.exists():
            try:
                from pycocotools.coco import COCO
                coco = COCO(str(ann_file))
                self.coco = coco
                self.cat_ids = coco.getCatIds()
                self.cat_id_to_name = {c["id"]: c["name"] for c in coco.loadCats(self.cat_ids)}
                img_ids = coco.getImgIds()
                self.img_ids = []
                for img_id in img_ids:
                    info = coco.loadImgs(img_id)[0]
                    fname = info["file_name"]
                    img_path = img_dir / fname
                    if img_path.exists():
                        self.image_paths.append(str(img_path))
                        self.label_paths.append(None)
                        self.img_ids.append(img_id)
                self.use_coco_api = True
                return
            except ImportError:
                self.coco = None
                self.use_coco_api = False

        self.coco = None
        self.use_coco_api = False

        # Fallback: no annotations, use all images with generic text
        for img_path in sorted(img_dir.glob("*.jpg")):
            self.image_paths.append(str(img_path))
            self.label_paths.append(None)
        self.coco = None
        self.img_ids = []
        self.use_coco_api = False

    def _get_class_names_yolo(self, label_path: Optional[str]) -> List[str]:
        """Get class names from YOLO label file."""
        if not label_path or not os.path.exists(label_path):
            return ["object"]
        class_ids = set()
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_ids.add(int(parts[0]))
        return [COCO_CLASS_NAMES[i] for i in sorted(class_ids) if 0 <= i < 80] or ["object"]

    def _get_class_names_coco(self, idx: int) -> List[str]:
        """Get class names from COCO annotations."""
        if not getattr(self, "use_coco_api", False) or not self.img_ids:
            return ["object"]
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        names = set()
        for ann in anns:
            cat_id = ann.get("category_id")
            if cat_id in self.cat_id_to_name:
                names.add(self.cat_id_to_name[cat_id])
        return list(names) if names else ["object"]

    def _make_text(self, class_names: List[str]) -> str:
        """Create CLIP-style text from class names."""
        if not class_names:
            return "a photo of an object"
        unique = list(dict.fromkeys(class_names))
        if len(unique) == 1:
            return f"a photo of a {unique[0]}"
        return "a photo of " + ", ".join(f"a {n}" for n in unique)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]

        if self.use_yolo_labels:
            class_names = self._get_class_names_yolo(self.label_paths[idx])
        else:
            class_names = self._get_class_names_coco(idx)

        text = self._make_text(class_names)
        text_tokens = self.tokenizer([text], truncate=True)[0]

        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        return img_tensor, text_tokens
