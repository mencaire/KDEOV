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

try:
    from torchvision.ops import box_iou, generalized_box_iou
except ImportError:
    box_iou = None
    generalized_box_iou = None

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


def box_best_cell(box_xyxy: torch.Tensor, default_boxes: torch.Tensor, hf: int, wf: int) -> int:
    """Return the cell index whose default box has the best IoU with the given box. Fallback: center cell if box_iou not available."""
    if box_iou is not None and default_boxes.device == box_xyxy.device:
        iou = box_iou(default_boxes, box_xyxy.unsqueeze(0)).squeeze(1)
        return iou.argmax().item()
    cx = (box_xyxy[0] + box_xyxy[2]) / 2
    cy = (box_xyxy[1] + box_xyxy[3]) / 2
    stride_w = IMAGE_SIZE / max(wf, 1)
    stride_h = IMAGE_SIZE / max(hf, 1)
    ix = min(int(cx / stride_w), wf - 1)
    iy = min(int(cy / stride_h), hf - 1)
    return iy * wf + ix


def train_finetune(
    checkpoint_path: str,
    data_root: str = "datasets",
    save_path: str = "checkpoints/kdeov_finetune",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-5,
    backbone: str = "yolov8n",
    fusion: str = "film",
    reg_weight: float = 2.0,
    neg_weight: float = 0.5,
    neg_margin: float = 0.0,
    max_neg_per_image: int = 32,
    use_best_iou_cell: bool = True,
    reg_loss: str = "smooth_l1",
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    num_classes = len(dataset.class_names)
    templates = ["a photo of a {}"]

    for epoch in range(epochs):
        total_loss = 0.0
        total_cls = 0.0
        total_reg = 0.0
        total_neg = 0.0
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
            logits = torch.matmul(spatial_flat, text_emb.t()) * model.detection_scale.exp()

            # Bbox regression: default boxes and predicted offsets
            default_boxes = grid_boxes_to_image(
                hf, wf, IMAGE_SIZE, IMAGE_SIZE, cell_scale=2.0, device=device
            )
            bbox_offsets = model.bbox_regression_head(fused_features)  # [B, 4, Hf, Wf]

            cls_loss_sum = 0.0
            reg_loss_sum = 0.0
            neg_loss_sum = 0.0
            count_cls = 0
            count_reg = 0
            count_neg = 0
            positive_cells_per_image = []
            for i in range(b):
                n = (labels[i] >= 0).sum().item()
                if n == 0:
                    positive_cells_per_image.append(set())
                    continue
                boxes_i = boxes[i][:n]
                labels_i = labels[i][:n]
                if use_best_iou_cell:
                    cell_indices = torch.tensor(
                        [box_best_cell(boxes_i[j], default_boxes, hf, wf) for j in range(n)],
                        device=boxes_i.device, dtype=torch.long
                    )
                else:
                    cell_indices = box_center_to_cell(boxes_i, hf, wf)
                positive_cells_per_image.append(set(cell_indices.cpu().tolist()))
                logits_i = logits[i]
                logits_at_cell = logits_i[cell_indices]
                cls_loss_sum += F.cross_entropy(logits_at_cell, labels_i)
                count_cls += 1
                for j in range(n):
                    cidx = cell_indices[j].item()
                    iy, ix = cidx // wf, cidx % wf
                    pred_offset = bbox_offsets[i, :, iy, ix]
                    pred_box = default_boxes[cidx] + pred_offset
                    target_box = boxes_i[j]
                    if reg_loss == "giou" and generalized_box_iou is not None:
                        giou = generalized_box_iou(pred_box.unsqueeze(0), target_box.unsqueeze(0)).squeeze()
                        reg_loss_sum += (1.0 - giou).clamp(min=0.0)
                    else:
                        target_offset = target_box - default_boxes[cidx]
                        reg_loss_sum += F.smooth_l1_loss(pred_offset, target_offset)
                    count_reg += 1
            # Negative (background) loss: push max logit down at cells with no object
            num_cells = hf * wf
            for i in range(b):
                pos_set = positive_cells_per_image[i]
                neg_candidates = [c for c in range(num_cells) if c not in pos_set]
                if len(neg_candidates) == 0:
                    continue
                n_neg = min(max_neg_per_image, len(neg_candidates))
                perm = torch.randperm(len(neg_candidates))[:n_neg].tolist()
                neg_indices = torch.tensor([neg_candidates[p] for p in perm], device=device, dtype=torch.long)
                max_logit_neg = logits[i][neg_indices].max(dim=1).values
                neg_loss_sum += F.relu(max_logit_neg - neg_margin).mean()
                count_neg += 1
            if count_cls == 0:
                continue
            cls_loss = cls_loss_sum / max(count_cls, 1)
            reg_loss = reg_loss_sum / max(count_reg, 1) if count_reg > 0 else torch.tensor(0.0, device=device)
            neg_loss = neg_loss_sum / max(count_neg, 1) if count_neg > 0 else torch.tensor(0.0, device=device)
            loss = cls_loss + reg_weight * reg_loss + neg_weight * neg_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            total_cls += cls_loss.item()
            total_reg += reg_loss.item() if count_reg > 0 else 0.0
            total_neg += neg_loss.item() if count_neg > 0 else 0.0
            num_batches += 1
            pbar.set_postfix(
                loss=loss.item(), cls=cls_loss.item(),
                reg=reg_loss.item() if count_reg > 0 else 0.0,
                neg=neg_loss.item() if count_neg > 0 else 0.0
            )

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        avg_cls = total_cls / max(num_batches, 1)
        avg_reg = total_reg / max(num_batches, 1)
        avg_neg = total_neg / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{epochs} avg loss: {avg_loss:.4f} (cls: {avg_cls:.4f}, reg: {avg_reg:.4f}, neg: {avg_neg:.6f}) lr: {scheduler.get_last_lr()[0]:.2e}")

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
    parser.add_argument("--reg-weight", type=float, default=2.0, help="Weight for bbox regression loss")
    parser.add_argument("--neg-weight", type=float, default=0.5, help="Weight for negative (background) cell loss")
    parser.add_argument("--neg-margin", type=float, default=0.0, help="Margin below which negative cell max-logit is not penalized")
    parser.add_argument("--max-neg-per-image", type=int, default=32, help="Max negative cells sampled per image")
    parser.add_argument("--no-best-iou-cell", action="store_true", help="Use center cell instead of best-IoU cell for GT assignment")
    parser.add_argument("--reg-loss", type=str, default="smooth_l1", choices=["smooth_l1", "giou"], help="Bbox regression loss: smooth_l1 or giou (better localization)")
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
        reg_weight=args.reg_weight,
        neg_weight=args.neg_weight,
        neg_margin=args.neg_margin,
        max_neg_per_image=args.max_neg_per_image,
        use_best_iou_cell=not args.no_best_iou_cell,
        reg_loss=args.reg_loss,
    )


if __name__ == "__main__":
    main()
