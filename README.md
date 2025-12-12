# Implementation-of-DDPM

```
Implementation-of-DDPM/
    ├── models/
    │     └── unet.py
    |     └── diffusion.py
    |     └── __init__.py 
    ├── scripts/
    │     └── train.py
    |     └── sample.py
    ├── notebooks/
    │     └── ddpm_from_scratch.ipynb
    ├── checkpoints/
    │     ├── unet_diffusion_trained.pt
    │     ├── ...
    ├── app/
    │     └── app.py
    ├── README
    ├── gitignore
    └── requirements.txt
```

## Dataset trained on

- **Used dataset**: `FashionMNIST` from `torchvision.datasets`.
- **Description**: 28×28 grayscale images of clothing items (10 classes).

- **Switching datasets**: to use the original `MNIST`, replace `FashionMNIST` with `MNIST` in the data-loading cells.

## SAMPLED IMAGES
### Samples after 5th epoch
<div align="center">
  <img src="sampled_images/epoch_5.png" width="300" />
</div>

### Samples after 100th epoch
<div align="center">
  <img src="sampled_images/epoch_100.png" width="300" />
</div>

