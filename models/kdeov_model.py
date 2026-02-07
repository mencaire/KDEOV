"""
KDEOV Main Model

Integrates all components:
- Frozen CLIP Text Encoder
- Lightweight Visual Backbone
- Projection Network
- Cross-Modal Fusion Module
- Spatial Projection (for open-vocabulary detection)

Supports:
- Zero-shot classification
- Text-image retrieval
- Open-vocabulary object detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union
import numpy as np

try:
    from torchvision.ops import nms as torch_nms
except ImportError:
    torch_nms = None

from .components import (
    FrozenCLIPTextEncoder,
    LightweightVisualBackbone,
    ProjectionNetwork,
    CrossModalFusionModule,
    SpatialProjection,
    grid_boxes_to_image,
)


class KDEOVModel(nn.Module):
    """
    Knowledge Distillation for Efficient Open-Vocabulary Vision Model
    
    A dual-stream, lightweight vision-language model that leverages:
    - Frozen CLIP model as semantic teacher
    - Lightweight YOLO-based backbone as student
    - End-to-end trainable with feature alignment pretraining
    """
    
    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        backbone_type: str = "yolov8n",
        fusion_type: str = "film",
        embedding_dim: int = 512,
        weights_dir: Optional[str] = None
    ):
        """
        Args:
            clip_model_name: CLIP model variant
            backbone_type: YOLO backbone type ("yolov8n" or "yolov5s")
            fusion_type: Fusion module type ("film" or "cross_attention")
            embedding_dim: Embedding dimension (should match CLIP)
            weights_dir: Directory to save/load backbone weights
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Get device
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
        
        # 1. Components Initialization
        self.text_encoder = FrozenCLIPTextEncoder(
            model_name=clip_model_name,
            device=str(self._device)
        )
        
        self.visual_backbone = LightweightVisualBackbone(
            backbone_type=backbone_type,
            pretrained=True,
            weights_dir=weights_dir
        )
        
        # Dynamically get actual feature dimension from backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_features = self.visual_backbone(dummy_input)
            # Handle list output (standard for YOLO)
            if isinstance(dummy_features, list):
                final_feature = dummy_features[-1]
            else:
                final_feature = dummy_features
            
            if len(final_feature.shape) > 2:
                backbone_output_dim = final_feature.shape[1]  # [B, C, H, W]
            else:
                backbone_output_dim = final_feature.shape[-1]  # [B, C]
        
        self.projection = ProjectionNetwork(
            input_dim=backbone_output_dim,
            output_dim=embedding_dim
        )
        
        self.fusion_module = CrossModalFusionModule(
            image_dim=backbone_output_dim,
            text_dim=embedding_dim,
            fusion_type=fusion_type
        )
        
        self.spatial_projection = SpatialProjection(
            input_dim=backbone_output_dim,
            output_dim=embedding_dim
        )
        
        # 2. CRITICAL: Add Temperature Parameter for Contrastive Learning
        # Initialized to log(1/0.07) following CLIP paper
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.clip_image_encoder = None
    
    def load_clip_image_encoder(self):
        """Load CLIP image encoder for distillation (frozen)"""
        if self.clip_image_encoder is None:
            import clip
            # Get device from model parameters
            device = next(self.parameters()).device
            clip_model, _ = clip.load(self.text_encoder.model_name, device=str(device))
            self.clip_image_encoder = clip_model.encode_image
            
            # Freeze
            for param in clip_model.parameters():
                param.requires_grad = False
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """
        Encode text to embeddings
        
        Args:
            text: Tokenized text [batch_size, seq_len]
        
        Returns:
            Text embeddings [batch_size, embedding_dim]
        """
        return self.text_encoder(text)
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to global embeddings (normalized).
        
        Args:
            images: Input images [batch_size, 3, H, W]
        
        Returns:
            Image embeddings [batch_size, embedding_dim]
        """
        # Extract features
        features = self.visual_backbone(images)
        
        # Use the final feature map if it's a list
        final_features = features[-1] if isinstance(features, list) else features
        
        # Project to embedding space [B, C, H, W]
        projected_map = self.projection(final_features)
        
        # Global Average Pooling: [B, C, H, W] -> [B, C]
        global_features = torch.mean(projected_map, dim=(2, 3))
        
        # L2 Normalize
        embeddings = F.normalize(global_features, dim=-1)
        
        return embeddings
    
    def forward(
        self, 
        images: torch.Tensor, 
        text_tokens: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Contrastive Learning and Detection.
        
        Args:
            images: Input images tensor [batch_size, 3, H, W]
            text_tokens: Tokenized text descriptions [batch_size, 77]
            
        Returns:
            Dictionary containing logits, features, and maps.
        """
        
        # 1. Visual Branch (Student)
        # ---------------------------------------------------------
        visual_features = self.visual_backbone(images)
        
        # Handle YOLO multi-scale output (take the deepest layer)
        if isinstance(visual_features, list):
            visual_features = visual_features[-1]
        
        # Project visual features [B, C, H, W]
        # This map is used for both global alignment and spatial detection
        projected_visual = self.projection(visual_features)
        
        outputs = {
            "visual_map": projected_visual # Keep spatial map for debugging/visualization
        }

        # 2. Text Branch & Contrastive Alignment
        # ---------------------------------------------------------
        if text_tokens is not None:
            # Get text embeddings [B, D]
            text_embeddings = self.text_encoder(text_tokens)
            
            # Normalize text embeddings
            text_embeddings = F.normalize(text_embeddings, dim=-1)
            outputs["text_features"] = text_embeddings

            # --- Global Contrastive Learning Path ---
            # Pool visual features to global vector [B, D]
            global_visual = torch.mean(projected_visual, dim=(2, 3))
            
            # Normalize visual embeddings
            global_visual = F.normalize(global_visual, dim=-1)
            outputs["visual_features"] = global_visual # Global vector
            
            # Calculate Logits (Similarity Scores)
            # logits = scale * (image @ text.T)
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * global_visual @ text_embeddings.t()
            logits_per_text = logits_per_image.t()
            
            outputs["logits"] = logits_per_image
            outputs["logits_per_text"] = logits_per_text

            # --- Spatial Detection Path (Optional Fusion) ---
            # Align data types for fusion (float16/float32 safety)
            if projected_visual.dtype != text_embeddings.dtype:
                text_embeddings = text_embeddings.to(projected_visual.dtype)
            
            # Fuse features for detection
            fused_map = self.fusion_module(projected_visual, text_embeddings)
            outputs["fused_map"] = fused_map
            
            # Predict boxes/scores if head exists
            if hasattr(self, 'spatial_projection'):
                predictions = self.spatial_projection(fused_map)
                outputs["predictions"] = predictions

        return outputs
    
    def compute_similarity(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between image and text embeddings
        
        Args:
            image_embeddings: [batch_size, embedding_dim] or [N, embedding_dim]
            text_embeddings: [batch_size, embedding_dim] or [M, embedding_dim]
        
        Returns:
            Similarity matrix [N, M] or [batch_size] if N==M
        """
        # Normalize
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        
        # Compute similarity
        if image_embeddings.shape[0] == text_embeddings.shape[0]:
            # Element-wise similarity
            similarity = (image_embeddings * text_embeddings).sum(dim=-1)
        else:
            # Matrix similarity
            similarity = torch.matmul(image_embeddings, text_embeddings.t())
        
        return similarity
    
    def zero_shot_classify(
        self,
        images: torch.Tensor,
        class_names: List[str],
        templates: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Zero-shot classification
        
        Args:
            images: Input images [batch_size, 3, H, W]
            class_names: List of class names
            templates: Optional prompt templates (default: ["a photo of a {}"])
        
        Returns:
            Logits [batch_size, num_classes]
        """
        import clip
        
        if templates is None:
            templates = ["a photo of a {}"]
        
        # Create prompts
        prompts = [template.format(name) for name in class_names for template in templates]
        
        # Tokenize
        tokenizer = clip.tokenize
        device = next(self.parameters()).device
        text_tokens = tokenizer(prompts).to(device)
        
        # Encode
        text_embeddings = self.text_encoder(text_tokens)
        image_embeddings = self.encode_image(images)
        
        # Average over templates if multiple
        if len(templates) > 1:
            num_classes = len(class_names)
            text_embeddings = text_embeddings.view(num_classes, len(templates), -1).mean(dim=1)
        
        # Compute similarity
        logits = self.compute_similarity(image_embeddings, text_embeddings)
        
        return logits

    def get_spatial_embeddings(
        self,
        images: torch.Tensor,
        text_embeddings: Optional[torch.Tensor] = None,
        use_fusion: bool = True,
    ) -> torch.Tensor:
        """
        Get per-location image embeddings for open-vocabulary detection.
        No global pooling; spatial dimensions are preserved.

        Args:
            images: [batch_size, 3, H, W]
            text_embeddings: Optional [batch_size, embedding_dim] or [1, embedding_dim]
                for text-guided fusion (broadcast over batch if needed).
            use_fusion: Whether to fuse with text when text_embeddings is provided.

        Returns:
            Spatial embeddings [batch_size, embedding_dim, Hf, Wf]
        """
        visual_features = self.visual_backbone(images)
        final_features = visual_features[-1] if isinstance(visual_features, list) else visual_features

        if text_embeddings is not None and use_fusion:
            # Broadcast text to batch if [1, D]
            if text_embeddings.shape[0] == 1 and final_features.shape[0] > 1:
                text_embeddings = text_embeddings.expand(final_features.shape[0], -1)
            final_features = self.fusion_module(final_features, text_embeddings)

        spatial_emb = self.spatial_projection(final_features)
        return spatial_emb

    def open_vocabulary_detect(
        self,
        images: torch.Tensor,
        class_names: List[str],
        templates: Optional[List[str]] = None,
        score_threshold: float = 0.2,
        nms_threshold: float = 0.5,
        cell_scale: float = 2.0,
        max_detections_per_image: int = 100,
    ) -> List[Dict[str, Union[torch.Tensor, List[str]]]]:
        """
        Open-vocabulary object detection: localize and classify with arbitrary text.

        Each spatial location in the feature map is scored against all class text
        embeddings; default boxes are generated from the grid and filtered by
        score and NMS.

        Args:
            images: Input images [batch_size, 3, H, W] (e.g. 224 or 640).
            class_names: Open-vocabulary class names (e.g. ["person", "car", "dog"]).
            templates: Prompt templates (default: ["a photo of a {}"]).
            score_threshold: Minimum similarity score to keep a detection.
            nms_threshold: IoU threshold for non-maximum suppression.
            cell_scale: Scale for default box size from grid cells.
            max_detections_per_image: Max detections to return per image.

        Returns:
            List of length batch_size. Each element is a dict:
            - "boxes": [N, 4] tensor in xyxy format (x1, y1, x2, y2)
            - "scores": [N] tensor
            - "labels": list of N class name strings
        """
        import clip

        if templates is None:
            templates = ["a photo of a {}"]
        device = next(self.parameters()).device
        batch_size, _, img_h, img_w = images.shape

        # Encode class names: one embedding per class (average over templates)
        prompts = [t.format(c) for c in class_names for t in templates]
        text_tokens = clip.tokenize(prompts, truncate=True).to(device)
        text_emb = self.text_encoder(text_tokens)
        if len(templates) > 1:
            num_classes = len(class_names)
            text_emb = text_emb.view(num_classes, len(templates), -1).mean(dim=1)
        else:
            text_emb = text_emb.view(len(class_names), -1)
        text_emb = F.normalize(text_emb, dim=-1)

        # Per-location image embeddings [B, D, Hf, Wf]
        spatial_emb = self.get_spatial_embeddings(images, text_emb, use_fusion=True)
        b, d, hf, wf = spatial_emb.shape
        num_cells = hf * wf

        # [B, Hf*Wf, D] @ [num_classes, D].t() -> [B, Hf*Wf, num_classes]
        spatial_flat = spatial_emb.permute(0, 2, 3, 1).reshape(b, num_cells, d)
        scores_all = torch.matmul(spatial_flat, text_emb.t())

        # Default boxes in image coordinates (xyxy)
        boxes = grid_boxes_to_image(hf, wf, img_h, img_w, cell_scale=cell_scale, device=device)
        boxes = boxes.unsqueeze(0).expand(batch_size, -1, -1)

        # Per cell: max score over classes -> objectness; argmax -> label
        cell_scores, label_indices = scores_all.max(dim=-1)
        label_indices = label_indices.cpu()

        out_list: List[Dict[str, Union[torch.Tensor, List[str]]]] = []
        for i in range(batch_size):
            scores_i = cell_scores[i]
            boxes_i = boxes[i]
            labels_i = label_indices[i]

            # Threshold
            keep = scores_i >= score_threshold
            if keep.sum() == 0:
                out_list.append({
                    "boxes": torch.zeros(0, 4, device=device, dtype=boxes.dtype),
                    "scores": torch.zeros(0, device=device),
                    "labels": [],
                })
                continue

            boxes_i = boxes_i[keep]
            scores_i = scores_i[keep]
            labels_i = labels_i[keep]
            label_names = [class_names[int(k)] for k in labels_i.tolist()]

            # NMS (per-image; class-agnostic) and cap number of detections
            if torch_nms is not None and len(boxes_i) > 1:
                keep_nms = torch_nms(boxes_i, scores_i, nms_threshold)
                keep_nms = keep_nms[:max_detections_per_image]
                boxes_i = boxes_i[keep_nms]
                scores_i = scores_i[keep_nms]
                label_names = [label_names[int(j)] for j in keep_nms.cpu().tolist()]
            elif len(boxes_i) > max_detections_per_image:
                top_scores, top_idx = scores_i.topk(max_detections_per_image, largest=True)
                boxes_i = boxes_i[top_idx]
                scores_i = top_scores
                label_names = [label_names[int(j)] for j in top_idx.cpu().tolist()]

            out_list.append({
                "boxes": boxes_i,
                "scores": scores_i,
                "labels": label_names,
            })

        return out_list

