"""
Model Components for KDEOV
Implements: Frozen CLIP Text Encoder, Lightweight Visual Backbone, 
Projection Network, and Cross-Modal Fusion Module
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import clip


class FrozenCLIPTextEncoder(nn.Module):
    """
    Frozen CLIP Text Encoder (Optimized)
    
    Uses the pretrained CLIP text encoder as a frozen semantic reference.
    
    Optimizations:
    1. Offline Loading: Checks local path first to prevent repeated downloads.
    2. Type Safety: Ensures output is float32 to match the visual backbone.
    3. Memory Efficiency: Explicitly disables gradient calculation.
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None, download_root: Optional[str] = None):
        """
        Args:
            model_name: CLIP model variant (e.g., "ViT-B/32", "ViT-L/14")
            device: Device to load CLIP model on (default: auto-detect - MPS/CUDA/CPU)
            download_root: Directory to store/load the pretrained model weights. If None, uses "./weights/clip".
        """
        super().__init__()
        self.model_name = model_name
        if download_root is None:
            download_root = os.path.join("weights", "clip")
        
        # 1. Device Auto-detection
        # Use provided device, or auto-detect (support Mac MPS, NVIDIA CUDA, or CPU)
        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        
        print(f"Initializing CLIP Text Encoder on device: {self.device}")

        # 2. Optimized Loading Logic (Offline Support)
        # Check if the model file already exists locally to avoid networking issues
        os.makedirs(download_root, exist_ok=True)
        model_filename = model_name.replace("/", "-") + ".pt"
        local_model_path = os.path.join(download_root, model_filename)

        try:
            if os.path.exists(local_model_path):
                print(f"Loading CLIP from local file: {local_model_path}")
                # Load from local file
                clip_model, _ = clip.load(local_model_path, device=self.device)
            else:
                print(f"Downloading CLIP to: {download_root}")
                # Download and load
                clip_model, _ = clip.load(model_name, device=self.device, download_root=download_root)
        except Exception as e:
            print(f"Error loading CLIP: {e}. Falling back to default download.")
            clip_model, _ = clip.load(model_name, device=self.device)

        # 3. Keep full CLIP model (needed for .parameters() and .encode_text)
        # We only use the text encoder in forward; visual part is not used but kept for parameter freezing
        self._clip_model = clip_model
        
        # Freeze all parameters to prevent updates during training (Knowledge Distillation)
        for param in self._clip_model.parameters():
            param.requires_grad = False
            
    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """
        Encodes text tokens into semantic embeddings.
        
        Args:
            text: Tokenized text tensor of shape [batch_size, 77]
            
        Returns:
            Tensor of shape [batch_size, embed_dim] (float32)
        """
        # Ensure no gradients are computed
        with torch.no_grad():
            # Encode text using CLIP
            embeddings = self._clip_model.encode_text(text)
            
            # Normalize embeddings (CLIP standard practice)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
            # CRITICAL: Cast to float32
            # CLIP usually outputs float16, but downstream components (YOLO) are often float32.
            # Mixing them can cause errors on MPS/CPU.
            return embeddings.float()

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
        input_size: int = 640,
        weights_dir: Optional[str] = None
    ):
        """
        Args:
            backbone_type: Type of YOLO backbone ("yolov8n" or "yolov5s")
            pretrained: Whether to use pretrained weights
            input_size: Input image size
            weights_dir: Directory to save/load YOLO weights (e.g. "weights"). If set, yolov8n.pt etc. are stored here.
        """
        super().__init__()
        self.backbone_type = backbone_type
        self.input_size = input_size
        self.use_yolo = False
        self.yolo_model = None
        self._hook_handles = []
        self._hook_features: List[torch.Tensor] = []
        self.feature_dims = None
        # Default to "weights" folder when pretrained so all checkpoints load from one place
        if pretrained and weights_dir is None:
            weights_dir = "weights"
        
        if backbone_type == "yolov8n":
            try:
                from ultralytics import YOLO
                if pretrained and weights_dir:
                    os.makedirs(weights_dir, exist_ok=True)
                    yolo_weights_path = os.path.join(weights_dir, "yolov8n.pt")
                else:
                    yolo_weights_path = "yolov8n.pt" if pretrained else None
                yolo_model = YOLO(yolo_weights_path)
                self.yolo_model = yolo_model.model
                # Ensure YOLO model parameters are trainable
                for param in self.yolo_model.parameters():
                    param.requires_grad = True
                self.use_yolo = True
                try:
                    self._register_yolo_hooks(self.yolo_model)
                except Exception as hook_error:
                    # If hook registration fails, fall back to simple backbone
                    print(f"Warning: Failed to register YOLO hooks: {hook_error}")
                    print("Falling back to simple CNN backbone")
                    self.use_yolo = False
                    self.yolo_model = None
                    self.backbone = self._create_simple_backbone()
                    self.feature_dims = [64, 128, 256, 512]
            except (ImportError, Exception) as e:
                # Fallback: Use a simplified CNN backbone
                print(f"Warning: Failed to load YOLOv8n: {e}")
                print("Falling back to simple CNN backbone")
                self.use_yolo = False
                self.yolo_model = None
                self.backbone = self._create_simple_backbone()
                self.feature_dims = [64, 128, 256, 512]
        elif backbone_type == "yolov5s":
            try:
                import torch.hub
                # yolov5s uses torch.hub; cache under weights/torch_hub when weights_dir is set
                if weights_dir:
                    torch_hub_dir = os.path.join(weights_dir, "torch_hub")
                    os.makedirs(torch_hub_dir, exist_ok=True)
                    os.environ["TORCH_HOME"] = torch_hub_dir
                yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=pretrained)
                self.yolo_model = yolo_model.model
                # Ensure YOLO model parameters are trainable
                for param in self.yolo_model.parameters():
                    param.requires_grad = True
                self.use_yolo = True
                try:
                    self._register_yolo_hooks(self.yolo_model)
                except Exception as hook_error:
                    # If hook registration fails, fall back to simple backbone
                    print(f"Warning: Failed to register YOLO hooks: {hook_error}")
                    print("Falling back to simple CNN backbone")
                    self.use_yolo = False
                    self.yolo_model = None
                    self.backbone = self._create_simple_backbone()
                    self.feature_dims = [64, 128, 256, 512]
            except Exception as e:
                print(f"Warning: Failed to load YOLOv5s: {e}")
                print("Falling back to simple CNN backbone")
                self.use_yolo = False
                self.yolo_model = None
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

    def _register_yolo_hooks(self, model: nn.Module) -> None:
        """
        Register forward hooks to capture intermediate feature maps from YOLO.
        Recursively registers hooks on Conv2d layers to capture multi-scale features.
        """
        def hook_fn(_module, _inputs, output):
            feature = self._extract_feature_tensor(output)
            if feature is not None:
                self._hook_features.append(feature)

        def register_hooks_recursive(module: nn.Module, depth: int = 0, max_depth: int = 10):
            """Recursively register hooks on Conv2d layers."""
            if depth > max_depth:
                return
            
            # Register hook on Conv2d layers (these produce feature maps)
            if isinstance(module, nn.Conv2d):
                self._hook_handles.append(module.register_forward_hook(hook_fn))
            
            # Recursively register on children
            for child in module.children():
                register_hooks_recursive(child, depth + 1, max_depth)
        
        # Start recursive registration from the model
        register_hooks_recursive(model, max_depth=15)
        
        # Also register on the model itself as fallback
        if len(self._hook_handles) == 0:
            self._hook_handles.append(model.register_forward_hook(hook_fn))

    def _extract_feature_tensor(self, output: object) -> Optional[torch.Tensor]:
        """
        Extract a 4D tensor feature map from a module output.
        """
        if isinstance(output, torch.Tensor) and output.dim() == 4:
            return output
        if isinstance(output, (list, tuple)):
            for item in reversed(output):
                if isinstance(item, torch.Tensor) and item.dim() == 4:
                    return item
        return None

    def _select_multiscale_features(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Select multi-scale features based on unique spatial resolutions.
        """
        size_to_feature = {}
        for feature in features:
            if not isinstance(feature, torch.Tensor) or feature.dim() != 4:
                continue
            size_to_feature[(feature.shape[2], feature.shape[3])] = feature
        if not size_to_feature:
            return []
        sorted_sizes = sorted(
            size_to_feature.keys(),
            key=lambda size: size[0] * size[1],
            reverse=True
        )
        return [size_to_feature[size] for size in sorted_sizes[:4]]
    
    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features from images
        
        Args:
            images: Input images [batch_size, 3, H, W]
            
        Returns:
            List of feature maps at different scales
        """
        if self.use_yolo and self.yolo_model is not None:
            try:
                self._hook_features = []
                _ = self.yolo_model(images)
                features = self._select_multiscale_features(self._hook_features)
                if not features:
                    # If no features captured, try to use the last hook feature
                    if len(self._hook_features) > 0:
                        features = [self._hook_features[-1]]
                    else:
                        # If still no features, fall back to simple backbone
                        # This shouldn't happen, but handle it gracefully
                        raise ValueError("No features captured from YOLO hooks")
                return features
            except Exception as e:
                # If YOLO forward fails, fall back to simple backbone
                print(f"Warning: YOLO forward pass failed: {e}")
                print("Falling back to simple CNN backbone")
                self.use_yolo = False
                # Continue to simple backbone processing below

        # Simple CNN backbone processing
        if not hasattr(self, 'backbone') or self.backbone is None:
            self.backbone = self._create_simple_backbone()
            self.feature_dims = [64, 128, 256, 512]
        
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
    Projection Network (Spatial-Aware)
    
    Maps visual features to CLIP embedding space while PRESERVING spatial dimensions.
    Uses 1x1 Convolutions instead of Linear layers to handle [B, C, H, W] inputs.
    
    Structure:
    - Conv2d 1x1 (Input -> Output)
    - BatchNorm2d
    - ReLU
    - Dropout
    - Conv2d 1x1 (Output -> Output)
    - BatchNorm2d
    """
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        """
        Args:
            input_dim: Input channel dimension (from backbone)
            output_dim: Output embedding dimension (e.g., 512 for CLIP)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.net = nn.Sequential(
            # 1. First Projection (Spatial Linear Map)
            nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_dim),  # BN is better for CNNs than LayerNorm
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # 2. Second Projection
            nn.Conv2d(output_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map [Batch, Input_Dim, H, W]
            
        Returns:
            Projected feature map [Batch, Output_Dim, H, W]
        """
        return self.net(x)


class SpatialProjection(nn.Module):
    """
    Spatial Projection for Open-Vocabulary Detection

    Projects backbone feature maps to CLIP embedding space while preserving
    spatial dimensions, so each location has an embedding for region-text similarity.
    Uses 1x1 conv + GroupNorm; no global pooling.
    """

    def __init__(self, input_dim: int, output_dim: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1),
            nn.GroupNorm(1, output_dim),  # normalize over channels per spatial location
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: [batch_size, C, H, W]

        Returns:
            [batch_size, output_dim, H, W], L2-normalized along channel dim
        """
        out = self.proj(image_features)
        # Normalize per spatial location (dim=1 is channel)
        out = F.normalize(out, dim=1)
        return out


def grid_boxes_to_image(
    feat_h: int,
    feat_w: int,
    img_h: int,
    img_w: int,
    cell_scale: float = 2.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Map feature grid indices to image-space boxes (xyxy format).

    Each cell (i, j) gets one default box: center at ((j+0.5)*stride_x, (i+0.5)*stride_y),
    size = (stride_x * cell_scale, stride_y * cell_scale). stride = image_size / feature_size.

    Args:
        feat_h, feat_w: Feature map height and width
        img_h, img_w: Image height and width
        cell_scale: Scale factor for default box size (e.g. 2.0 = overlapping cells)
        device: Device for the output tensor

    Returns:
        Boxes tensor [feat_h * feat_w, 4] in xyxy format (x1, y1, x2, y2)
    """
    stride_y = img_h / max(feat_h, 1)
    stride_x = img_w / max(feat_w, 1)
    half_h = (stride_y * cell_scale) / 2
    half_w = (stride_x * cell_scale) / 2
    ys = torch.arange(feat_h, dtype=torch.float32, device=device)
    xs = torch.arange(feat_w, dtype=torch.float32, device=device)
    cy = (ys + 0.5) * stride_y
    cx = (xs + 0.5) * stride_x
    grid_cy, grid_cx = torch.meshgrid(cy, cx, indexing="ij")
    # clamp centers to image
    grid_cy = grid_cy.clamp(0, img_h - 1e-4)
    grid_cx = grid_cx.clamp(0, img_w - 1e-4)
    x1 = (grid_cx - half_w).clamp(0, img_w)
    x2 = (grid_cx + half_w).clamp(0, img_w)
    y1 = (grid_cy - half_h).clamp(0, img_h)
    y2 = (grid_cy + half_h).clamp(0, img_h)
    boxes = torch.stack([x1, y1, x2, y2], dim=-1).view(-1, 4)
    return boxes


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
        
        if scale.shape[-1] != actual_image_dim:
            raise ValueError(
                "FiLM dimension mismatch: expected image feature channels "
                f"{scale.shape[-1]}, got {actual_image_dim}. "
                "Ensure image_dim matches backbone output channels."
            )
        
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

