"""
Training Script for Feature Alignment Pretraining

This script demonstrates how to train the KDEOV model using:
- Distillation Loss (student vs teacher CLIP embeddings)
- Cross-Modal Alignment Loss (image-text contrastive loss)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
import os
import clip  # type: ignore

from models import KDEOVModel, FeatureAlignmentLoss


def train_feature_alignment(
    model: KDEOVModel,
    dataloader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    save_path: Optional[str] = None
):
    """
    Train model with feature alignment pretraining
    
    Args:
        model: KDEOV model instance (assumed to be on CUDA)
        dataloader: DataLoader with (images, texts) pairs
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        save_path: Path to save checkpoint
    """
    # Load CLIP image encoder for teacher embeddings
    model.load_clip_image_encoder()
    
    # Loss function
    criterion = FeatureAlignmentLoss(
        distillation_weight=1.0,
        alignment_weight=1.0,
        distillation_type="cosine",
        temperature=0.07
    )
    
    # Optimizer (only train student components, not frozen CLIP)
    trainable_params = [
        param for param in model.parameters() if param.requires_grad
    ]
    optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_dist_loss = 0.0
        total_align_loss = 0.0
        
        for batch_idx, (images, texts) in enumerate(dataloader):
            images = images.cuda()
            texts = texts.cuda()
            
            # Forward pass through student model
            student_outputs = model(images=images, text=texts, use_fusion=True)
            student_image_embeddings = student_outputs["image_embeddings"]
            
            # Get teacher embeddings from frozen CLIP image encoder
            with torch.no_grad():
                teacher_image_embeddings = model.clip_image_encoder(images)
                teacher_image_embeddings = torch.nn.functional.normalize(
                    teacher_image_embeddings, dim=-1
                )
            
            # Get text embeddings
            text_embeddings = student_outputs["text_embeddings"]
            
            # Compute loss
            loss_dict = criterion(
                student_image_embeddings=student_image_embeddings,
                teacher_image_embeddings=teacher_image_embeddings,
                text_embeddings=text_embeddings
            )
            
            loss = loss_dict["total_loss"]
            dist_loss = loss_dict["distillation_loss"]
            align_loss = loss_dict["alignment_loss"]
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_dist_loss += dist_loss.item()
            total_align_loss += align_loss.item()
            
            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Dist: {dist_loss.item():.4f}, "
                    f"Align: {align_loss.item():.4f}"
                )
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch summary
        avg_loss = total_loss / len(dataloader)
        avg_dist = total_dist_loss / len(dataloader)
        avg_align = total_align_loss / len(dataloader)
        
        print(
            f"\nEpoch {epoch+1} Summary:\n"
            f"  Average Total Loss: {avg_loss:.4f}\n"
            f"  Average Distillation Loss: {avg_dist:.4f}\n"
            f"  Average Alignment Loss: {avg_align:.4f}\n"
        )
        
        # Save checkpoint
        if save_path:
            # Create directory if it doesn't exist
            checkpoint_dir = os.path.dirname(save_path)
            if checkpoint_dir and not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"{save_path}_epoch_{epoch+1}.pt")
    
    print("Training completed!")


if __name__ == "__main__":
    # Example usage
    # Initialize model (automatically on CUDA)
    model = KDEOVModel(
        clip_model_name="ViT-B/32",
        backbone_type="yolov8n",
        fusion_type="film"
    ).cuda()
    
    # Example: Create a dummy dataset
    # In practice, you would use a real dataset with image-text pairs
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=1000):
            self.num_samples = num_samples
            self.tokenizer = clip.tokenize
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Dummy image (3, 224, 224)
            image = torch.randn(3, 224, 224)
            # Dummy text
            text = "a photo of a cat"
            text_tokens = self.tokenizer([text])[0]
            return image, text_tokens
    
    # Create dataloader
    dataset = DummyDataset(num_samples=1000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train
    print("Starting feature alignment pretraining...")
    train_feature_alignment(
        model=model,
        dataloader=dataloader,
        num_epochs=5,
        learning_rate=1e-4,
        save_path="checkpoints/kdeov"
    )

