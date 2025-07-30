import numpy as np
import torch

def generate_spiral_data(n_samples=2048, noise_std=0.05):
    theta = np.sqrt(np.random.rand(n_samples)) * 2 * np.pi
    r = 2 * theta
    x = r * np.cos(theta) + noise_std * np.random.randn(n_samples)
    y = r * np.sin(theta) + noise_std * np.random.randn(n_samples)
    data = np.stack([x, y], axis=1)
    return torch.tensor(data, dtype=torch.float32)
