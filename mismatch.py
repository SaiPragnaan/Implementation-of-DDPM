import torch
import sys
import os

# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.diffusion import DiffusionModel

DEVICE = "cpu"
CHECKPOINT_PATH = "checkpoints/unet_diffusion_ema_final.pt"

def check_mismatch():
    model = DiffusionModel(T=200, in_channels=1, out_channels=1)
    
    try:
        sd = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    except FileNotFoundError:
        print("Checkpoint not found.")
        return

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(sd.keys())

    # 1. Keys in your code but NOT in the file (These are random noise!)
    missing_in_file = model_keys - ckpt_keys
    
    # 2. Shape mismatches
    print("--- Checking Shape Mismatches ---")
    mismatches = []
    for k in model_keys.intersection(ckpt_keys):
        model_shape = model.state_dict()[k].shape
        ckpt_shape = sd[k].shape
        if model_shape != ckpt_shape:
            mismatches.append(f"{k}: Code={model_shape} vs File={ckpt_shape}")
            
    if len(mismatches) > 0:
        print("CRITICAL: The following layers have different shapes and were NOT loaded:")
        for m in mismatches:
            print(m)
    else:
        print("No shape mismatches found.")

    print("\n--- Checking Missing Keys ---")
    if len(missing_in_file) > 0:
        print("CRITICAL: The following layers exist in your code but NOT in the checkpoint:")
        for k in missing_in_file:
            # We ignore the buffer keys (betas, alphas) as those are expected
            if "beta" not in k and "alpha" not in k:
                print(k)
    else:
        print("All layers present.")

if __name__ == "__main__":
    check_mismatch()