"""
Loss Functions for Feature Alignment Pretraining

Implements:
- Distillation Loss: Aligns student image embeddings with teacher CLIP embeddings
- Cross-Modal Alignment Loss: Ensures image and text embeddings are semantically aligned
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DistillationLoss(nn.Module):
    """
    Distillation Loss
    
    Transfers semantic richness from large CLIP model to compact student model.
    Can use Cosine Embedding Loss or L2 Loss between projected image embeddings
    and frozen CLIP image encoder embeddings.
    """
    
    def __init__(
        self,
        loss_type: str = "cosine",  # "cosine" or "l2"
        temperature: float = 0.07,
        reduction: str = "mean"
    ):
        """
        Args:
            loss_type: Type of loss ("cosine" or "l2")
            temperature: Temperature scaling for cosine similarity
            reduction: Reduction method ("mean", "sum", or "none")
        """
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        self.reduction = reduction
        
        if loss_type == "cosine":
            self.cosine_sim = nn.CosineSimilarity(dim=-1)
        elif loss_type == "l2":
            self.mse_loss = nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distillation loss
        
        Args:
            student_embeddings: Projected image embeddings from student model
                               [batch_size, embedding_dim]
            teacher_embeddings: Image embeddings from frozen CLIP image encoder
                               [batch_size, embedding_dim]
        
        Returns:
            Distillation loss value
        """
        # Ensure dtype consistency
        if student_embeddings.dtype != teacher_embeddings.dtype:
            # Convert to the higher precision dtype (float32)
            target_dtype = torch.float32 if student_embeddings.dtype == torch.float32 or teacher_embeddings.dtype == torch.float32 else teacher_embeddings.dtype
            student_embeddings = student_embeddings.to(dtype=target_dtype)
            teacher_embeddings = teacher_embeddings.to(dtype=target_dtype)
        
        # Ensure embeddings are normalized
        student_embeddings = F.normalize(student_embeddings, dim=-1)
        teacher_embeddings = F.normalize(teacher_embeddings, dim=-1)
        
        if self.loss_type == "cosine":
            # Cosine similarity loss
            # Maximize similarity (minimize 1 - similarity)
            cosine_sim = self.cosine_sim(student_embeddings, teacher_embeddings)
            loss = 1 - cosine_sim
            
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            else:
                return loss
        
        else:  # l2
            # L2/MSE loss
            return self.mse_loss(student_embeddings, teacher_embeddings)


class CrossModalAlignmentLoss(nn.Module):
    """
    Cross-Modal Alignment Loss
    
    Uses contrastive loss (InfoNCE) to ensure image and text embeddings
    are semantically aligned in the shared embedding space.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        reduction: str = "mean"
    ):
        """
        Args:
            temperature: Temperature scaling for logits
            reduction: Reduction method ("mean", "sum", or "none")
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive alignment loss (InfoNCE)
        
        Args:
            image_embeddings: Image embeddings [batch_size, embedding_dim]
            text_embeddings: Text embeddings [batch_size, embedding_dim]
            labels: Optional labels for positive pairs [batch_size]
                   If None, assumes diagonal pairs are positive
        
        Returns:
            Contrastive loss value
        """
        batch_size = image_embeddings.shape[0]
        
        # Ensure dtype consistency
        if image_embeddings.dtype != text_embeddings.dtype:
            # Convert to the higher precision dtype (float32)
            target_dtype = torch.float32 if image_embeddings.dtype == torch.float32 or text_embeddings.dtype == torch.float32 else text_embeddings.dtype
            image_embeddings = image_embeddings.to(dtype=target_dtype)
            text_embeddings = text_embeddings.to(dtype=target_dtype)
        
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        
        # Compute similarity matrix
        # [batch_size, batch_size]
        similarity_matrix = torch.matmul(image_embeddings, text_embeddings.t()) / self.temperature
        
        # Create labels (diagonal = positive pairs)
        if labels is None:
            labels = torch.arange(batch_size, device=image_embeddings.device)
        else:
            # Use provided labels to create positive pair mask
            labels = labels.long()
        
        # InfoNCE loss: -log(exp(pos) / sum(exp(all)))
        # For each image, the corresponding text is the positive pair
        loss = F.cross_entropy(similarity_matrix, labels, reduction=self.reduction)
        
        return loss


class FeatureAlignmentLoss(nn.Module):
    """
    Combined Feature Alignment Loss
    
    Combines distillation loss and cross-modal alignment loss for
    end-to-end training of the feature alignment pretraining stage.
    """
    
    def __init__(
        self,
        distillation_weight: float = 1.0,
        alignment_weight: float = 1.0,
        distillation_type: str = "cosine",
        temperature: float = 0.07
    ):
        """
        Args:
            distillation_weight: Weight for distillation loss
            alignment_weight: Weight for cross-modal alignment loss
            distillation_type: Type of distillation loss ("cosine" or "l2")
            temperature: Temperature scaling
        """
        super().__init__()
        self.distillation_weight = distillation_weight
        self.alignment_weight = alignment_weight
        
        self.distillation_loss = DistillationLoss(
            loss_type=distillation_type,
            temperature=temperature
        )
        self.alignment_loss = CrossModalAlignmentLoss(
            temperature=temperature
        )
    
    def forward(
        self,
        student_image_embeddings: torch.Tensor,
        teacher_image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Compute combined feature alignment loss
        
        Args:
            student_image_embeddings: Projected image embeddings from student
                                     [batch_size, embedding_dim]
            teacher_image_embeddings: Image embeddings from frozen CLIP
                                     [batch_size, embedding_dim]
            text_embeddings: Text embeddings [batch_size, embedding_dim]
            labels: Optional labels for positive pairs
        
        Returns:
            Dictionary with individual and total losses
        """
        # Distillation loss
        dist_loss = self.distillation_loss(
            student_image_embeddings,
            teacher_image_embeddings
        )
        
        # Cross-modal alignment loss
        align_loss = self.alignment_loss(
            student_image_embeddings,
            text_embeddings,
            labels
        )
        
        # Combined loss
        total_loss = (
            self.distillation_weight * dist_loss +
            self.alignment_weight * align_loss
        )
        
        return {
            "total_loss": total_loss,
            "distillation_loss": dist_loss,
            "alignment_loss": align_loss
        }

