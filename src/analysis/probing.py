import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)

def train_probe(features, labels, epochs=10, lr=1e-3, device='cuda'):
    """
    Trains a linear probe on frozen features to predict a specific target.
    This is used to test object vs. background separability at different layers.
    
    Args:
        features (Tensor): Frozen extracted representations from a specific ViT layer.
        labels (Tensor): Target labels (either object labels or background labels).
        
    Returns:
        float: Accuracy of the linear probe on a held-out validation set.
    """
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    probe = LinearProbe(features.shape[1], len(torch.unique(labels))).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # TODO: Implement training loop and train/val split.
    # ...
    
    return probe
