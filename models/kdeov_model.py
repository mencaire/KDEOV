"""
KDEOV Main Model

Integrates all components:
- Frozen CLIP Text Encoder
- Lightweight Visual Backbone
- Projection Network
- Cross-Modal Fusion Module

Supports:
- Zero-shot classification
- Text-image retrieval
- Open-vocabulary object detection (future)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict

from .components import (
    FrozenCLIPTextEncoder,
    LightweightVisualBackbone,
    ProjectionNetwork,
    CrossModalFusionModule
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
        embedding_dim: int = 512
    ):
        """
        Args:
            clip_model_name: CLIP model variant
            backbone_type: YOLO backbone type ("yolov8n" or "yolov5s")
            fusion_type: Fusion module type ("film" or "cross_attention")
            embedding_dim: Embedding dimension (should match CLIP)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Get device (will be set when model is moved to device)
        # For now, default to CUDA if available
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Components
        self.text_encoder = FrozenCLIPTextEncoder(
            model_name=clip_model_name,
            device=str(self._device)
        )
        
        self.visual_backbone = LightweightVisualBackbone(
            backbone_type=backbone_type,
            pretrained=True
        )
        
        # Dynamically get actual feature dimension from backbone
        # Create a dummy input to get real output dimensions
        # Note: Keep dummy_input on CPU since model is not moved to CUDA yet during __init__
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)  # Keep on CPU
            dummy_features = self.visual_backbone(dummy_input)
            # Get the last feature map
            if isinstance(dummy_features, list):
                final_feature = dummy_features[-1]
            else:
                final_feature = dummy_features
            # Get channel dimension
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
        
        # For training: we'll need CLIP image encoder for distillation
        # This will be loaded separately when needed
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
        Encode images to embeddings
        
        Args:
            images: Input images [batch_size, 3, H, W]
        
        Returns:
            Image embeddings [batch_size, embedding_dim]
        """
        # Extract features
        features = self.visual_backbone(images)
        
        # Use the final feature map
        final_features = features[-1] if isinstance(features, list) else features
        
        # Project to embedding space
        embeddings = self.projection(final_features)
        
        return embeddings
    
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        use_fusion: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            images: Input images [batch_size, 3, H, W]
            text: Tokenized text [batch_size, seq_len]
            use_fusion: Whether to use cross-modal fusion
        
        Returns:
            Dictionary with embeddings and/or similarity scores
        """
        outputs = {}
        
        if images is not None:
            # Extract visual features
            visual_features = self.visual_backbone(images)
            final_features = visual_features[-1] if isinstance(visual_features, list) else visual_features
            
            # Apply fusion if text is provided
            if text is not None and use_fusion:
                text_embeddings = self.text_encoder(text)
                fused_features = self.fusion_module(final_features, text_embeddings)
                # Project fused features
                image_embeddings = self.projection(fused_features)
            else:
                # Project without fusion
                image_embeddings = self.projection(final_features)
            
            outputs["image_embeddings"] = image_embeddings
        
        if text is not None:
            text_embeddings = self.text_encoder(text)
            outputs["text_embeddings"] = text_embeddings
        
        # Compute similarity if both are provided
        if images is not None and text is not None:
            image_emb = outputs["image_embeddings"]
            text_emb = outputs["text_embeddings"]
            
            # Cosine similarity
            similarity = F.cosine_similarity(image_emb, text_emb, dim=-1)
            outputs["similarity"] = similarity
        
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

