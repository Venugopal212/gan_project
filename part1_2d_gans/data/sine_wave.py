import numpy as np
import torch

def generate_sine_wave_data(n_samples=2048, noise_std=0.05):
    x = np.random.uniform(-3, 3, n_samples)
    y = np.sin(2 * x) + noise_std * np.random.randn(n_samples)
    data = np.stack([x, y], axis=1)
    return torch.tensor(data, dtype=torch.float32)
