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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from models import KDEOVModel, FeatureAlignmentLoss


def plot_training_curves(
    loss_history: Dict[str, list],
    save_path: Optional[str] = None,
    epoch: Optional[int] = None
):
    """
    Plot training curves for loss visualization
    
    Args:
        loss_history: Dictionary with 'total', 'distillation', 'alignment' keys
        save_path: Path to save the plot (without extension)
        epoch: Current epoch number
    """
    if len(loss_history['total']) == 0:
        return
    
    epochs = np.arange(1, len(loss_history['total']) + 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    # Plot 1: All losses together
    ax1 = axes[0]
    ax1.plot(epochs, loss_history['total'], 'b-o', label='Total Loss', linewidth=2, markersize=6)
    ax1.plot(epochs, loss_history['distillation'], 'g-s', label='Distillation Loss', linewidth=1.5, markersize=5, alpha=0.7)
    ax1.plot(epochs, loss_history['alignment'], 'r-^', label='Alignment Loss', linewidth=1.5, markersize=5, alpha=0.7)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Losses', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0.5, right=len(epochs) + 0.5)
    
    # Highlight best loss
    if len(loss_history['total']) > 0:
        best_idx = np.argmin(loss_history['total'])
        best_epoch = best_idx + 1
        best_loss = loss_history['total'][best_idx]
        ax1.plot(best_epoch, best_loss, 'r*', markersize=15, label=f'Best: {best_loss:.4f}')
        ax1.annotate(f'Best: {best_loss:.4f}', 
                    xy=(best_epoch, best_loss),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                    fontsize=9)
    
    # Plot 2: Total loss with trend line
    ax2 = axes[1]
    ax2.plot(epochs, loss_history['total'], 'b-o', linewidth=2, markersize=8, label='Total Loss')
    
    # Add trend line (moving average)
    if len(loss_history['total']) >= 3:
        window = min(3, len(loss_history['total']))
        trend = np.convolve(loss_history['total'], np.ones(window)/window, mode='valid')
        trend_epochs = np.arange(window, len(loss_history['total']) + 1)
        ax2.plot(trend_epochs, trend, 'r--', linewidth=2, alpha=0.7, label=f'{window}-epoch Moving Average')
    
    # Mark current epoch
    if epoch is not None and epoch < len(epochs):
        ax2.plot(epoch + 1, loss_history['total'][epoch], 'go', markersize=12, label='Current Epoch')
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Total Loss', fontsize=12)
    ax2.set_title('Total Loss Trend', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0.5, right=len(epochs) + 0.5)
    
    # Add convergence indicator
    if len(loss_history['total']) >= 3:
        recent_losses = loss_history['total'][-3:]
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        cv = (loss_std / loss_mean) * 100 if loss_mean > 0 else 0
        
        if cv < 1.0:
            status_text = "Status: CONVERGED"
            status_color = 'green'
        elif cv < 3.0:
            status_text = "Status: STABILIZING"
            status_color = 'orange'
        else:
            status_text = "Status: TRAINING"
            status_color = 'blue'
        
        ax2.text(0.02, 0.98, status_text, 
                transform=ax2.transAxes, 
                fontsize=11, fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.2))
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        plot_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
        if plot_dir and not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)
        
        plot_filename = f"{save_path}_training_curves.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"  Training curves saved to: {plot_filename}")
    
    plt.close(fig)


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
    
    # Track loss history for convergence analysis
    loss_history = {
        'total': [],
        'distillation': [],
        'alignment': []
    }
    best_loss = float('inf')
    
    print("=" * 80)
    print("Training Configuration:")
    print(f"  Total Epochs: {num_epochs}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Total Batches per Epoch: {len(dataloader)}")
    print(f"  Batch Size: {dataloader.batch_size}")
    print("=" * 80)
    print()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_dist_loss = 0.0
        total_align_loss = 0.0
        num_batches = 0
        
        # Track batch losses for this epoch
        batch_losses = []
        
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
            loss_val = loss.item()
            dist_loss_val = dist_loss.item()
            align_loss_val = align_loss.item()
            
            total_loss += loss_val
            total_dist_loss += dist_loss_val
            total_align_loss += align_loss_val
            batch_losses.append(loss_val)
            num_batches += 1
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                progress = (batch_idx + 1) / len(dataloader) * 100
                print(
                    f"Epoch {epoch+1}/{num_epochs} [{progress:5.1f}%] | "
                    f"Batch {batch_idx:4d}/{len(dataloader)-1} | "
                    f"Loss: {loss_val:6.4f} | "
                    f"Dist: {dist_loss_val:6.4f} | "
                    f"Align: {align_loss_val:6.4f} | "
                    f"LR: {current_lr:.2e}"
                )
        
        # Update learning rate
        scheduler.step()
        
        # Calculate average losses
        avg_loss = total_loss / num_batches
        avg_dist = total_dist_loss / num_batches
        avg_align = total_align_loss / num_batches
        
        # Store in history
        loss_history['total'].append(avg_loss)
        loss_history['distillation'].append(avg_dist)
        loss_history['alignment'].append(avg_align)
        
        # Check for best loss
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        
        # Calculate loss change and trend
        if epoch > 0:
            loss_change = avg_loss - loss_history['total'][epoch-1]
            loss_change_pct = (loss_change / loss_history['total'][epoch-1]) * 100
            trend = "DOWN" if loss_change < 0 else "UP"
        else:
            loss_change = 0.0
            loss_change_pct = 0.0
            trend = "-"
        
        # Convergence analysis
        if epoch >= 2:
            recent_losses = loss_history['total'][-3:]
            loss_std = torch.tensor(recent_losses).std().item()
            loss_mean = torch.tensor(recent_losses).mean().item()
            cv = (loss_std / loss_mean) * 100 if loss_mean > 0 else 0  # Coefficient of variation
            
            if cv < 1.0:
                convergence_status = "CONVERGED"
            elif cv < 3.0:
                convergence_status = "STABILIZING"
            else:
                convergence_status = "TRAINING"
        else:
            convergence_status = "TRAINING"
        
        # Print detailed epoch summary
        print()
        print("=" * 80)
        print(f"Epoch {epoch+1}/{num_epochs} Summary")
        print("=" * 80)
        print(f"  Total Loss:      {avg_loss:8.4f}  {trend}  ({loss_change:+.4f}, {loss_change_pct:+.2f}%)")
        print(f"  Distillation:    {avg_dist:8.4f}")
        print(f"  Alignment:       {avg_align:8.4f}")
        print(f"  Learning Rate:   {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Status:          {convergence_status}")
        if is_best:
            print(f"  Best Loss:       {best_loss:8.4f} (NEW BEST!)")
        else:
            print(f"  Best Loss:       {best_loss:8.4f}")
        
        # Show loss trend over last few epochs
        if epoch >= 1:
            print(f"\n  Loss Trend (last {min(5, epoch+1)} epochs):")
            start_idx = max(0, epoch - 4)
            for i in range(start_idx, epoch + 1):
                marker = "*" if i == epoch else " "
                print(f"    Epoch {i+1:2d}: {loss_history['total'][i]:8.4f} {marker}")
        
        print("=" * 80)
        print()
        
        # Plot and save training curves
        plot_training_curves(loss_history, save_path, epoch)
        
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
    
    # Final training summary
    print("=" * 80)
    print("Training Completed!")
    print("=" * 80)
    print(f"  Total Epochs: {num_epochs}")
    print(f"  Best Loss: {best_loss:.4f} (Epoch {loss_history['total'].index(best_loss) + 1})")
    print(f"  Final Loss: {loss_history['total'][-1]:.4f}")
    
    # Calculate overall improvement
    if len(loss_history['total']) > 1:
        initial_loss = loss_history['total'][0]
        final_loss = loss_history['total'][-1]
        improvement = ((initial_loss - final_loss) / initial_loss) * 100
        print(f"  Improvement: {improvement:.2f}% (from {initial_loss:.4f} to {final_loss:.4f})")
    
    # Final convergence status
    if len(loss_history['total']) >= 3:
        recent_losses = loss_history['total'][-3:]
        loss_std = torch.tensor(recent_losses).std().item()
        loss_mean = torch.tensor(recent_losses).mean().item()
        cv = (loss_std / loss_mean) * 100 if loss_mean > 0 else 0
        
        if cv < 1.0:
            print(f"  Convergence: CONVERGED (CV: {cv:.2f}%)")
        elif cv < 3.0:
            print(f"  Convergence: STABILIZING (CV: {cv:.2f}%)")
        else:
            print(f"  Convergence: STILL TRAINING (CV: {cv:.2f}%)")
    
    print("=" * 80)
    
    # Save final training curves
    if save_path:
        plot_training_curves(loss_history, save_path, epoch=num_epochs-1)
        print(f"\nFinal training curves saved!")


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

