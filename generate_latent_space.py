import torch
from config import load_config

config = load_config('config.yaml')

def generate_latent_points(latent_dim, n_samples):
    return torch.randn(n_samples, latent_dim, device=config['DEVICE'])