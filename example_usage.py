"""
Example Usage of KDEOV Model

Demonstrates:
- Zero-shot classification
- Text-image retrieval
- Model inference
"""

import torch
import clip
from PIL import Image
import requests
from io import BytesIO

from models import KDEOVModel


def example_zero_shot_classification():
    """Example: Zero-shot image classification"""
    print("=" * 50)
    print("Example 1: Zero-shot Classification")
    print("=" * 50)
    
    # Initialize model (automatically on CUDA)
    model = KDEOVModel(
        clip_model_name="ViT-B/32",
        backbone_type="yolov8n",
        fusion_type="film"
    ).cuda()
    model.eval()
    
    # Example image (in practice, load from file)
    image = torch.randn(1, 3, 224, 224).cuda()
    
    # Class names
    class_names = ["cat", "dog", "bird", "car", "bicycle"]
    
    # Classify
    with torch.no_grad():
        logits = model.zero_shot_classify(image, class_names)
        probs = torch.softmax(logits, dim=-1)
    
    # Print results
    print(f"\nClassification Results:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {probs[0][i].item():.4f}")
    
    predicted_class = class_names[torch.argmax(probs, dim=-1).item()]
    print(f"\nPredicted: {predicted_class}")


def example_text_image_retrieval():
    """Example: Text-image retrieval"""
    print("\n" + "=" * 50)
    print("Example 2: Text-Image Retrieval")
    print("=" * 50)
    
    # Initialize model (automatically on CUDA)
    model = KDEOVModel(
        clip_model_name="ViT-B/32",
        backbone_type="yolov8n",
        fusion_type="film"
    ).cuda()
    model.eval()
    
    # Example: Database of images
    num_images = 10
    images = torch.randn(num_images, 3, 224, 224).cuda()
    
    # Query text
    query_text = "a photo of a cat"
    tokenizer = clip.tokenize
    text_tokens = tokenizer([query_text]).cuda()
    
    # Encode images and text
    with torch.no_grad():
        image_embeddings = model.encode_image(images)
        text_embeddings = model.encode_text(text_tokens)
    
    # Compute similarity
    similarities = model.compute_similarity(image_embeddings, text_embeddings)
    
    # Get top-k most similar images
    k = 3
    top_k_indices = torch.topk(similarities, k=k, dim=0).indices.squeeze()
    
    print(f"\nQuery: '{query_text}'")
    print(f"\nTop-{k} most similar images:")
    for i, idx in enumerate(top_k_indices):
        print(f"  Rank {i+1}: Image {idx.item()} (similarity: {similarities[idx].item():.4f})")


def example_forward_pass():
    """Example: Standard forward pass"""
    print("\n" + "=" * 50)
    print("Example 3: Forward Pass")
    print("=" * 50)
    
    # Initialize model (automatically on CUDA)
    model = KDEOVModel(
        clip_model_name="ViT-B/32",
        backbone_type="yolov8n",
        fusion_type="film"
    ).cuda()
    model.eval()
    
    # Example inputs
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224).cuda()
    query_text = "a photo of a dog"
    tokenizer = clip.tokenize
    text_tokens = tokenizer([query_text] * batch_size).cuda()
    
    # Forward pass
    with torch.no_grad():
        outputs = model(images=images, text=text_tokens, use_fusion=True)
    
    print(f"\nInput shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Text tokens: {text_tokens.shape}")
    
    print(f"\nOutput shapes:")
    print(f"  Image embeddings: {outputs['image_embeddings'].shape}")
    print(f"  Text embeddings: {outputs['text_embeddings'].shape}")
    print(f"  Similarity: {outputs['similarity'].shape}")
    
    print(f"\nSimilarity scores:")
    for i, sim in enumerate(outputs['similarity']):
        print(f"  Image {i}: {sim.item():.4f}")


if __name__ == "__main__":
    print("KDEOV Model - Example Usage\n")
    
    # Run examples
    try:
        example_zero_shot_classification()
        example_text_image_retrieval()
        example_forward_pass()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("=" * 50)
    except Exception as e:
        print(f"\nError: {e}")
        print("Note: Some examples may require actual model weights or data.")
        print("This is a demonstration of the API structure.")

