import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.diffusion import DiffusionModel


def get_train_loader(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    return loader

class EMA:
    def __init__(self, model: nn.Module, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 100 
LR = 2e-4
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "unet_diffusion_trained.pt")

def train():
    print("Initializing Data Loader...")
    train_loader = get_train_loader(BATCH_SIZE)
    
    model = DiffusionModel(T=1000, in_channels=1, out_channels=1).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    
    ema = EMA(model)

    model.train()
    print(f"Starting training on {DEVICE}...")

    losses = []
    
    for epoch in tqdm(range(EPOCHS)):
        total_loss_of_epoch = 0.0
        
        for batch_X, _ in train_loader:
            batch_X = batch_X.to(DEVICE)
            B = batch_X.shape[0]

            t = torch.randint(1, model.T + 1, (B,)).to(DEVICE)
            eps_preds, eps = model(batch_X, t) 
            
            loss = loss_fn(eps_preds, eps)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            ema.update(model)
            total_loss_of_epoch += loss.item()
        
        avg_loss = total_loss_of_epoch / len(train_loader)
        print(f"Epoch : {epoch+1}/{EPOCHS} -- Loss : {avg_loss:.4f}")
        losses.append(avg_loss)


    print("Training complete. Swapping to EMA weights for saving...")
    ema.apply_shadow(model)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Final EMA Model saved to {CHECKPOINT_PATH}")

if __name__ == "__main__":
    train()