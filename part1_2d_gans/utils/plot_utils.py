import matplotlib.pyplot as plt
import torch

def plot_2d_scatter(real, fake, title, save_path=None):
    plt.figure(figsize=(6, 6))
    real_np = real.detach().cpu().numpy()
    fake_np = fake.detach().cpu().numpy()
    plt.scatter(real_np[:, 0], real_np[:, 1], alpha=0.5, label="Real", s=10)
    plt.scatter(fake_np[:, 0], fake_np[:, 1], alpha=0.5, label="Fake", s=10)
    plt.legend()
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()
