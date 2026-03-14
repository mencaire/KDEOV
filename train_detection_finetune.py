"""
Detection fine-tuning: train with bounding box labels after feature alignment.

Uses COCO train2017 (bbox from instances_train2017.json). For each GT box,
the cell containing its center is trained to predict the correct class
(spatial feature · text_embedding). No new data download if you have coco_lvis.

Usage:
  python train_detection_finetune.py --checkpoint checkpoints/kdeov_coco_lvis_epoch_10.pt \\
    --epochs 5 --batch-size 16 --save-path checkpoints/kdeov_finetune
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import clip
from tqdm import tqdm

from models import KDEOVModel
from models.components import grid_boxes_to_image
from data.detection_dataset import COCODetectionDataset, collate_detection

IMAGE_SIZE = 224


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def box_center_to_cell(boxes_xyxy: torch.Tensor, hf: int, wf: int) -> torch.Tensor:
    """Map box centers (in 224 coords) to flat cell index in [0, hf*wf)."""
    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
    stride_w = IMAGE_SIZE / max(wf, 1)
    stride_h = IMAGE_SIZE / max(hf, 1)
    ix = (cx / stride_w).long().clamp(0, wf - 1)
    iy = (cy / stride_h).long().clamp(0, hf - 1)
    return (iy * wf + ix).long()


def train_finetune(
    checkpoint_path: str,
    data_root: str = "datasets",
    save_path: str = "checkpoints/kdeov_finetune",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-5,
    backbone: str = "yolov8n",
    fusion: str = "film",
):
    device = get_device()
    print(f"Device: {device}")

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = KDEOVModel(
        clip_model_name="ViT-B/32",
        backbone_type=backbone,
        fusion_type=fusion,
        weights_dir="weights",
    )
    load_ret = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if load_ret.missing_keys:
        print(f"Note: loading with strict=False. Missing keys (e.g. bbox_regression_head): {load_ret.missing_keys[:5]}{'...' if len(load_ret.missing_keys) > 5 else ''}")
    if load_ret.unexpected_keys:
        print(f"Unexpected keys in checkpoint: {load_ret.unexpected_keys[:5]}...")
    model = model.to(device)
    model.train()

    dataset = COCODetectionDataset(data_root=data_root)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_detection,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_classes = len(dataset.class_names)
    templates = ["a photo of a {}"]

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, boxes, labels in pbar:
            images = images.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)
            b = images.size(0)

            with torch.no_grad():
                prompts = [t.format(c) for c in dataset.class_names for t in templates]
                text_tokens = clip.tokenize(prompts, truncate=True).to(device)
                text_emb = model.text_encoder(text_tokens)
                text_emb = text_emb.view(num_classes, -1)
                text_emb = F.normalize(text_emb, dim=-1)
                text_emb_fusion = text_emb.mean(dim=0, keepdim=True)

            spatial_emb, fused_features = model.get_spatial_embeddings_and_fused(
                images, text_emb_fusion, use_fusion=True
            )
            b, d, hf, wf = spatial_emb.shape
            num_cells = hf * wf
            spatial_flat = spatial_emb.permute(0, 2, 3, 1).reshape(b, num_cells, d)
            logits = torch.matmul(spatial_flat, text_emb.t())

            # Bbox regression: default boxes and predicted offsets
            default_boxes = grid_boxes_to_image(
                hf, wf, IMAGE_SIZE, IMAGE_SIZE, cell_scale=2.0, device=device
            )
            bbox_offsets = model.bbox_regression_head(fused_features)  # [B, 4, Hf, Wf]

            cls_loss_sum = 0.0
            reg_loss_sum = 0.0
            count_cls = 0
            count_reg = 0
            for i in range(b):
                n = (labels[i] >= 0).sum().item()
                if n == 0:
                    continue
                boxes_i = boxes[i][:n]
                labels_i = labels[i][:n]
                cell_indices = box_center_to_cell(boxes_i, hf, wf)
                logits_i = logits[i]
                logits_at_cell = logits_i[cell_indices]
                cls_loss_sum += F.cross_entropy(logits_at_cell, labels_i)
                count_cls += 1
                for j in range(n):
                    cidx = cell_indices[j].item()
                    iy, ix = cidx // wf, cidx % wf
                    target_offset = boxes_i[j] - default_boxes[cidx].to(device)
                    pred_offset = bbox_offsets[i, :, iy, ix]
                    reg_loss_sum += F.smooth_l1_loss(pred_offset, target_offset)
                    count_reg += 1
            if count_cls == 0:
                continue
            cls_loss = cls_loss_sum / max(count_cls, 1)
            reg_loss = reg_loss_sum / max(count_reg, 1) if count_reg > 0 else torch.tensor(0.0, device=device)
            reg_weight = 2.0
            loss = cls_loss + reg_weight * reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=loss.item(), cls=cls_loss.item(), reg=reg_loss.item() if count_reg > 0 else 0.0)

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{epochs} avg loss: {avg_loss:.4f}")

    Path(save_path).mkdir(parents=True, exist_ok=True)
    out_file = Path(save_path) / "kdeov_finetune.pt"
    torch.save(
        {"epoch": epochs, "model_state_dict": model.state_dict()},
        out_file,
    )
    print(f"Saved fine-tuned checkpoint to {out_file}")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Detection fine-tuning with bbox labels")
    parser.add_argument("--checkpoint", type=str, required=True, help="Feature-alignment .pt checkpoint")
    parser.add_argument("--data-root", type=str, default="datasets")
    parser.add_argument("--save-path", type=str, default="checkpoints/kdeov_finetune")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--backbone", type=str, default="yolov8n", choices=["yolov8n", "yolov5s"])
    parser.add_argument("--fusion", type=str, default="film", choices=["film", "cross_attention"])
    args = parser.parse_args()

    train_finetune(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone=args.backbone,
        fusion=args.fusion,
    )


if __name__ == "__main__":
    main()
