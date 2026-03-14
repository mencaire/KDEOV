"""
LVIS Dataset for KDEOV Feature Alignment Training (Open-Vocabulary Object Detection)

LVIS uses COCO 2017 images. Requires:
- datasets/coco2017/train2017/, val2017/
- datasets/lvis/annotations/lvis_v1_train.json, lvis_v1_val.json

Usage:
  split="train" -> train2017 + lvis_v1_train.json (for training)
  split="val"   -> val2017 + lvis_v1_val.json (for evaluation)
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import clip  # type: ignore

IMAGE_SIZE = 224


class LVISDataset(Dataset):
    """
    LVIS dataset for image-text pair training.
    Uses COCO 2017 images + LVIS annotations (1,203 categories).
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        coco_root: Optional[str] = None,
        tokenizer=None,
    ):
        """
        Args:
            data_root: Root directory (e.g. "datasets")
            split: "train" or "val"
            coco_root: Override COCO path (default: data_root/coco2017)
            tokenizer: CLIP tokenizer
        """
        self.data_root = Path(data_root)
        self.split = split
        self.coco_root = Path(coco_root) if coco_root else self.data_root / "coco2017"
        self.tokenizer = tokenizer or clip.tokenize

        self._load_annotations()
        self._build_image_index()

    def _load_annotations(self) -> None:
        ann_dir = self.data_root / "lvis" / "annotations"
        ann_file = ann_dir / f"lvis_v1_{self.split}.json"
        if not ann_file.exists():
            raise FileNotFoundError(
                f"LVIS annotations not found: {ann_file}\n"
                f"Run: python download_data.py --dataset lvis"
            )

        with open(ann_file) as f:
            data = json.load(f)

        self.images = {img["id"]: img for img in data["images"]}
        self.categories: Dict[int, str] = {}
        for cat in data["categories"]:
            # LVIS uses "synonyms" (list), take first as name
            name = cat.get("name") or (cat.get("synonyms", ["object"])[0])
            self.categories[cat["id"]] = name

        # image_id -> list of category names
        self.img_to_cats: Dict[int, List[str]] = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            cat_id = ann["category_id"]
            if cat_id in self.categories:
                name = self.categories[cat_id]
                if img_id not in self.img_to_cats:
                    self.img_to_cats[img_id] = []
                if name not in self.img_to_cats[img_id]:
                    self.img_to_cats[img_id].append(name)

    def _build_image_index(self) -> None:
        img_dir = self.coco_root / ("train2017" if self.split == "train" else "val2017")
        if not img_dir.exists():
            raise FileNotFoundError(
                f"COCO images not found: {img_dir}\n"
                f"Run: python download_data.py --dataset coco2017 --parts train2017 val2017"
            )

        self.image_paths: List[str] = []
        self.img_ids: List[int] = []
        for img_id, img_info in self.images.items():
            # LVIS JSON may have "file_name" or only "coco_url" (e.g. http://.../train2017/000000123.jpg)
            fname = img_info.get("file_name")
            if fname is None and "coco_url" in img_info:
                fname = img_info["coco_url"].rstrip("/").split("/")[-1]
            if not fname:
                continue
            path = img_dir / fname
            if path.exists():
                self.image_paths.append(str(path))
                self.img_ids.append(img_id)

    def _make_text(self, class_names: List[str]) -> str:
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
        img_id = self.img_ids[idx]
        class_names = self.img_to_cats.get(img_id, ["object"])
        text = self._make_text(class_names)
        text_tokens = self.tokenizer([text], truncate=True)[0]

        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        return img_tensor, text_tokens
