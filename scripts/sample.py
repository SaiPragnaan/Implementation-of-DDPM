import torch
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.diffusion import DiffusionModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/unet_diffusion_ema_final.pt"

IMG_SHAPE = (1, 1, 28, 28) 

def save_image(tensor, filename="generated_sample.png"):
    tensor = tensor.detach().cpu()
    
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    if tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
        
    plt.imsave(filename, tensor.numpy(), cmap='gray')
    print(f"Image saved successfully to: {os.path.abspath(filename)}")

def sample():
    print(f"Loading model on {DEVICE}...")

    model = DiffusionModel(T=200, in_channels=1, out_channels=1, beta_schedule='cosine').to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Checkpoint loaded.")
    else:
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Please run train.py first.")
        return

    model.eval()
    
    print("Sampling new image (this may take a moment)...")
    with torch.no_grad():
        generated_images = model.generate_output(IMG_SHAPE)

    save_image(generated_images[0], "fashion_mnist_sample.png")

if __name__ == "__main__":
    sample()