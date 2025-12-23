"""
Model Components for KDEOV
Implements: Frozen CLIP Text Encoder, Lightweight Visual Backbone, 
Projection Network, and Cross-Modal Fusion Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import clip


class FrozenCLIPTextEncoder(nn.Module):
    """
    Frozen CLIP Text Encoder
    
    Uses the pretrained CLIP text encoder as a frozen semantic reference.
    The text encoder processes text prompts and outputs semantic embeddings
    in the CLIP embedding space.
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        """
        Args:
            model_name: CLIP model variant (e.g., "ViT-B/32", "ViT-L/14")
            device: Device to load CLIP model on (default: "cuda" if available, else "cpu")
        """
        super().__init__()
        self.model_name = model_name
        
        # 自动检测设备 (支持 Mac MPS, NVIDIA CUDA, 普通 CPU)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        print(f"Loading CLIP Text Encoder on: {self.device}")
        
        # Load CLIP model on the correct device
        clip_model, _ = clip.load(model_name, device=self.device)
        self.text_encoder = clip_model.encode_text
        
        # Freeze all parameters
        for param in clip_model.parameters():
            param.requires_grad = False
        
        self.clip_model = clip_model
        self.embedding_dim = clip_model.text_projection.shape[1]
        
    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """
        Encode text to embeddings
        
        Args:
            text: Tokenized text input [batch_size, seq_len]
            
        Returns:
            Text embeddings [batch_size, embedding_dim]
        """
        with torch.no_grad():
            text_features = self.text_encoder(text)
            # Normalize embeddings
            text_features = F.normalize(text_features, dim=-1)
        return text_features
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Alias for forward"""
        return self.forward(text)


class LightweightVisualBackbone(nn.Module):
    """
    Lightweight Visual Backbone
    
    Uses YOLO backbone (YOLOv5s or YOLOv8n) for efficient feature extraction.
    Extracts multi-scale features suitable for object-level representation.
    """
    
    def __init__(
        self,
        backbone_type: str = "yolov8n",  # or "yolov5s"
        pretrained: bool = True,
        input_size: int = 640
    ):
        """
        Args:
            backbone_type: Type of YOLO backbone ("yolov8n" or "yolov5s")
            pretrained: Whether to use pretrained weights
            input_size: Input image size
        """
        super().__init__()
        self.backbone_type = backbone_type
        self.input_size = input_size
        
        if backbone_type == "yolov8n":
            try:
                from ultralytics import YOLO
                yolo_model = YOLO('yolov8n.pt' if pretrained else None)
                # Extract backbone (first part of the model)
                self.backbone = yolo_model.model.model[:10]  # Adjust based on actual architecture
                self.feature_dims = [64, 128, 256, 512]  # Typical YOLOv8 feature dimensions
            except ImportError:
                # Fallback: Use a simplified CNN backbone
                self.backbone = self._create_simple_backbone()
                self.feature_dims = [64, 128, 256, 512]
        elif backbone_type == "yolov5s":
            try:
                import torch.hub
                yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=pretrained)
                self.backbone = yolo_model.model.model[:23]  # Extract backbone layers
                self.feature_dims = [64, 128, 256, 512]
            except Exception:
                self.backbone = self._create_simple_backbone()
                self.feature_dims = [64, 128, 256, 512]
        else:
            # Default: Simple CNN backbone
            self.backbone = self._create_simple_backbone()
            self.feature_dims = [64, 128, 256, 512]
    
    def _create_simple_backbone(self) -> nn.Module:
        """
        Create a simple CNN backbone as fallback
        Mimics YOLO-style feature extraction
        """
        return nn.Sequential(
            # Initial conv block
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 1
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features from images
        
        Args:
            images: Input images [batch_size, 3, H, W]
            
        Returns:
            List of feature maps at different scales
        """
        features = []
        x = images
        
        # Extract features at different scales
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            # Collect features at key points
            if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)) and len(features) < 4:
                if x.shape[1] in self.feature_dims:
                    features.append(x)
        
        # Ensure we have multi-scale features
        if len(features) < 2:
            # Use the final feature map
            features = [x]
        
        return features


class ProjectionNetwork(nn.Module):
    """
    Projection Network
    
    Maps image features from the lightweight backbone to the CLIP embedding space.
    Uses a 2-layer MLP with normalization.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 512,  # CLIP embedding dimension
        hidden_dim: Optional[int] = None
    ):
        """
        Args:
            input_dim: Dimension of input image features
            output_dim: Dimension of output embeddings (CLIP embedding dim)
            hidden_dim: Hidden layer dimension (default: input_dim)
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Project image features to CLIP embedding space
        
        Args:
            image_features: Image features [batch_size, ..., input_dim]
            
        Returns:
            Projected embeddings [batch_size, ..., output_dim]
        """
        # Flatten spatial dimensions if needed
        original_shape = image_features.shape
        if len(original_shape) > 2:
            # Global average pooling
            image_features = F.adaptive_avg_pool2d(image_features, (1, 1))
            image_features = image_features.view(original_shape[0], -1)
        
        # Project to embedding space
        embeddings = self.projection(image_features)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1)
        
        return embeddings


class CrossModalFusionModule(nn.Module):
    """
    Cross-Modal Fusion Module
    
    Implements Feature-wise Linear Modulation (FiLM) to fuse text embeddings
    with image features. Text embeddings modulate intermediate feature maps
    of the visual backbone, enabling text-guided visual processing.
    """
    
    def __init__(
        self,
        image_dim: int = 512,
        text_dim: int = 512,
        fusion_type: str = "film"  # "film" or "cross_attention"
    ):
        """
        Args:
            image_dim: Dimension of image features
            text_dim: Dimension of text embeddings
            fusion_type: Type of fusion ("film" or "cross_attention")
        """
        super().__init__()
        self.fusion_type = fusion_type
        self.image_dim = image_dim
        self.text_dim = text_dim
        
        if fusion_type == "film":
            # FiLM: Feature-wise Linear Modulation
            # Generate scale and shift parameters from text
            self.film_generator = nn.Sequential(
                nn.Linear(text_dim, image_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(image_dim * 2, image_dim * 2)
            )
        elif fusion_type == "cross_attention":
            # Cross-attention mechanism
            self.query_proj = nn.Linear(image_dim, image_dim)
            self.key_proj = nn.Linear(text_dim, image_dim)
            self.value_proj = nn.Linear(text_dim, image_dim)
            self.output_proj = nn.Linear(image_dim, image_dim)
            self.scale = (image_dim) ** -0.5
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse image features with text embeddings
        
        Args:
            image_features: Image features [batch_size, ..., image_dim]
            text_embeddings: Text embeddings [batch_size, text_dim]
            
        Returns:
            Fused features [batch_size, ..., image_dim]
        """
        if self.fusion_type == "film":
            return self._film_fusion(image_features, text_embeddings)
        else:
            return self._cross_attention_fusion(image_features, text_embeddings)
    
    def _film_fusion(
        self,
        image_features: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Feature-wise Linear Modulation (FiLM)
        
        Modulates image features using scale and shift parameters
        generated from text embeddings.
        """
        # Ensure dtype consistency: convert text_embeddings to match film_generator dtype
        if text_embeddings.dtype != next(self.film_generator.parameters()).dtype:
            text_embeddings = text_embeddings.to(dtype=next(self.film_generator.parameters()).dtype)
        
        # Get actual image feature channel dimension
        if len(image_features.shape) > 2:
            actual_image_dim = image_features.shape[1]  # [B, C, H, W]
        else:
            actual_image_dim = image_features.shape[-1]  # [B, C]
        
        # Generate FiLM parameters (scale and shift)
        film_params = self.film_generator(text_embeddings)
        scale, shift = torch.chunk(film_params, 2, dim=-1)
        
        # If dimensions don't match, create or use projection layers
        if scale.shape[-1] != actual_image_dim:
            # Create projection layers if they don't exist or dimensions changed
            proj_key = f'_film_proj_{scale.shape[-1]}_to_{actual_image_dim}'
            if not hasattr(self, proj_key):
                # Create and register as buffer/parameter so it's part of the model
                scale_proj = nn.Linear(scale.shape[-1], actual_image_dim).to(scale.device).to(scale.dtype)
                shift_proj = nn.Linear(shift.shape[-1], actual_image_dim).to(shift.device).to(shift.dtype)
                # Register as submodules so they're part of the model
                self.add_module(f'film_scale_proj_{scale.shape[-1]}_{actual_image_dim}', scale_proj)
                self.add_module(f'film_shift_proj_{shift.shape[-1]}_{actual_image_dim}', shift_proj)
                setattr(self, proj_key, (scale_proj, shift_proj))
            
            scale_proj, shift_proj = getattr(self, proj_key)
            scale = scale_proj(scale)
            shift = shift_proj(shift)
        
        # Expand to match image feature spatial dimensions
        original_shape = image_features.shape
        if len(original_shape) > 2:
            # For spatial features, expand scale and shift
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            shift = shift.unsqueeze(-1).unsqueeze(-1)
        else:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        
        # Apply FiLM: output = scale * features + shift
        fused_features = scale * image_features + shift
        
        return fused_features
    
    def _cross_attention_fusion(
        self,
        image_features: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-attention fusion
        
        Uses text embeddings as keys/values and image features as queries.
        """
        # Ensure dtype consistency: convert text_embeddings to match projection layer dtype
        if text_embeddings.dtype != next(self.key_proj.parameters()).dtype:
            text_embeddings = text_embeddings.to(dtype=next(self.key_proj.parameters()).dtype)
        
        # Flatten spatial dimensions if needed
        original_shape = image_features.shape
        if len(original_shape) > 2:
            batch_size, channels, height, width = original_shape
            image_features_flat = image_features.view(batch_size, channels, -1).transpose(1, 2)
        else:
            image_features_flat = image_features.unsqueeze(1)
        
        # Project to query, key, value
        queries = self.query_proj(image_features_flat)
        keys = self.key_proj(text_embeddings).unsqueeze(1)
        values = self.value_proj(text_embeddings).unsqueeze(1)
        
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        attended_features = torch.matmul(attention_weights, values)
        attended_features = self.output_proj(attended_features)
        
        # Reshape back if needed
        if len(original_shape) > 2:
            attended_features = attended_features.transpose(1, 2).view(original_shape)
        else:
            attended_features = attended_features.squeeze(1)
        
        # Residual connection
        fused_features = image_features + attended_features
        
        return fused_features

