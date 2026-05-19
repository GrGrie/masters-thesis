import torch
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

def compute_attention_distance(attention_weights, patch_size=16, image_size=224):
    """
    Computes the mean attention distance as described in Dosovitskiy et al. and Park et al.
    This metric shows whether the layer aggregates information locally or globally.
    
    Args:
        attention_weights (Tensor): Attention maps of shape (B, num_heads, num_tokens, num_tokens).
        patch_size (int): Size of the patches (e.g., 16).
        image_size (int): Image dimension (e.g., 224).
        
    Returns:
        float: The mean attention distance for this layer.
    """
    # TODO: Implement grid distance computation between tokens and weight them by the attention map.
    # Needs to handle the [CLS] token separately as it has no spatial position.
    pass

def compute_attention_homogeneity(attention_weights):
    """
    Measures how similar attention maps of different query tokens are.
    High homogeneity means many tokens look at the exact same things (less diverse).
    
    Args:
        attention_weights (Tensor): Attention maps of shape (B, num_heads, num_tokens, num_tokens).
        
    Returns:
        float: Homogeneity score (e.g., based on Normalized Mutual Information).
    """
    # TODO: Flatten spatial dimensions and compute pairwise NMI or cosine similarity 
    # between query tokens' attention distributions.
    pass
