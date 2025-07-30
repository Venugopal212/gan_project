import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=2, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, z):
        return self.net(z)
