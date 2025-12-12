
import streamlit as st
import torch
import sys
import os
import numpy as np

# Add the root directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.diffusion import DiffusionModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/unet_diffusion_ema_final.pt"
IMG_SHAPE = (1, 1, 28, 28)

def tensor_to_image(tensor):
    tensor = tensor.detach().cpu()
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor.numpy()

st.title("Fashion MNIST Image Generation")
st.write("Click the button below to generate a new image.")

if st.button("Generate Image"):
    st.write("Loading model...")
    model = DiffusionModel(T=200, in_channels=1, out_channels=1,beta_schedule='cosine').to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        st.write("Checkpoint loaded.")
    else:
        st.error(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        st.error("Please run train.py first.")
        st.stop()

    model.eval()

    st.write("Sampling new image (this may take a moment)...")
    with torch.no_grad():
        generated_images = model.generate_output(IMG_SHAPE)

    img_np = tensor_to_image(generated_images[0])

    st.image(img_np, caption="Generated Image", use_column_width=True, channels="GRAY")
