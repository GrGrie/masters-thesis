import torch
import numpy as np
import matplotlib.pyplot as plt
import umap

def compute_singular_values(features):
    """
    Computes the singular values of the feature matrix to measure spectral diversity.
    A rapidly decaying spectrum indicates representation collapse.
    
    Args:
        features (Tensor): Feature matrix of shape (N, D).
        
    Returns:
        Tensor: Singular values.
    """
    # Center the features
    features_centered = features - features.mean(dim=0, keepdim=True)
    
    # Compute SVD
    U, S, V = torch.svd(features_centered)
    return S

def compute_effective_rank(singular_values):
    """
    Computes the effective rank (Shannon entropy of the normalized singular values).
    
    Args:
        singular_values (Tensor): 1D tensor of singular values.
        
    Returns:
        float: Effective rank.
    """
    p = singular_values / singular_values.sum()
    entropy = -torch.sum(p * torch.log(p + 1e-8))
    return torch.exp(entropy).item()

def plot_umap(features, labels, title="UMAP Projection", save_path=None):
    """
    Projects high-dimensional features to 2D using UMAP and plots them.
    
    Args:
        features (numpy.ndarray): Feature matrix (N, D).
        labels (numpy.ndarray): Labels for coloring points.
        title (str): Title of the plot.
        save_path (str): Path to save the plot.
    """
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    embedding = reducer.fit_transform(features)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=5)
    plt.colorbar(scatter)
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()
