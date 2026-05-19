import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Runs one epoch of fine-tuning.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # For Waterbirds: image, target (bird type), background (place)
        images, targets, backgrounds = batch
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    acc = 100. * correct / total
    return total_loss / len(dataloader), acc

def train_model(model, train_loader, val_loader, config, device='cuda'):
    """
    Main training loop for fine-tuning the model.
    """
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], 
                            weight_decay=config['training']['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    # Initialize Weights & Biases if configured
    # wandb.init(project="masters-thesis", config=config, name=config['experiment']['name'])
    
    best_val_acc = 0.0
    
    for epoch in range(config['training']['epochs']):
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validation
        # val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_loss, val_acc = 0.0, 0.0 # Placeholder
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        # wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
        
        # Save best model
        # if val_acc > best_val_acc:
        #    best_val_acc = val_acc
        #    torch.save(model.state_dict(), f"{config['experiment']['output_dir']}/best_model.pth")
            
    # wandb.finish()
