import torch
import torch.nn as nn
import math
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.unet import UNet

def cosine_beta_schedule(T, s=0.008):
    steps = T
    timesteps = torch.arange(steps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((timesteps / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = []
    for t in range(1, steps + 1):
        beta = min(1 - (alphas_cumprod[t] / alphas_cumprod[t - 1]), 0.999)
        betas.append(beta)
    return torch.tensor(betas, dtype=torch.float32)

class DiffusionModel(nn.Module):
    def __init__(self, T=1000, in_channels=1, out_channels=1, beta_schedule='linear'):
        super().__init__()
        self.unet = UNet(dimension=128, in_channels=in_channels, out_channels=out_channels)

        self.T = T
        if beta_schedule == 'cosine':
            betas = cosine_beta_schedule(self.T)
        else:
            betas = torch.linspace(1e-4, 0.02, T)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1.0 - betas)
        self.register_buffer('alpha_bars', torch.cumprod(1.0 - betas, dim=0))
        alpha_bars = torch.cumprod(1.0 - betas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]], dim=0)
        self.register_buffer('alpha_bars_prev', alpha_bars_prev)

        self.register_buffer('sqrt_alpha_bars', torch.sqrt(self.alpha_bars))
        self.register_buffer('sqrt_om_alpha_bars', torch.sqrt(1.0 - self.alpha_bars))
 

    def q_sample(self,x0,t,eps):
        # x0: [B,C,H,W] ,t: [B] ints in [1..T]
        sqrt_alpha_t_bar=self.sqrt_alpha_bars[t-1].view(-1,1,1,1)
        om_sqrt_alpha_t_bar=self.sqrt_om_alpha_bars[t-1].view(-1,1,1,1)
        xt=sqrt_alpha_t_bar*x0 + om_sqrt_alpha_t_bar*eps

        return xt
    
    def p_mean_variance(self, xt, t):
        # xt: [B,C,H,W], t: [B] in [1..T]
        epsilon_hat = self.unet(xt, t)
        
        device = xt.device
        t_idx = (t - 1)  # 0-based indexing
        alpha_t = self.alphas[t_idx].view(-1, 1, 1, 1).to(device)
        beta_t = self.betas[t_idx].view(-1, 1, 1, 1).to(device)
        alpha_bar_t = self.alpha_bars[t_idx].view(-1, 1, 1, 1).to(device)
        alpha_bar_prev = self.alpha_bars_prev[t_idx].view(-1, 1, 1, 1).to(device)
        sqrt_om_alpha_t_bar = torch.sqrt(1.0 - alpha_bar_t) 

        mu_theta = (1.0 / torch.sqrt(alpha_t)) * (xt - (beta_t / sqrt_om_alpha_t_bar) * epsilon_hat)

        posterior_variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)

        return mu_theta, posterior_variance, epsilon_hat

    def p_sample(self, xt, t):
        mu, var, eps_hat = self.p_mean_variance(xt, t)

        is_final = (t == 1).all()
        if is_final:
            return mu

        z = torch.randn_like(xt)
        sigma = torch.sqrt(var)
        return mu + sigma * z


    def generate_output(self, shape, device="cpu"):
        xt = torch.randn(shape, device=device)
    
        for t in reversed(range(1, self.T + 1)):
            t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)
            xt = self.p_sample(xt, t_batch)
    
        return xt

    def forward(self,x0,t):
        # x0: [B,C,H,W] ,t: [B] 
        eps=torch.randn_like(x0)  # we need epsilon also of size  [B,C,H,W]
        xts=self.q_sample(x0,t,eps) # xts: [B,C,H,W]
        eps_pred = self.unet(xts, t)

        return eps_pred,eps