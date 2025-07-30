import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from gan.dcgan_module import DCGAN
from gan.cgan_module import CGAN
import argparse
import os
import glob
import warnings

warnings.filterwarnings("ignore")


def save_image_grid(images, filename, nrow=8, normalize=True):
    grid = make_grid(images, nrow=nrow, normalize=normalize, pad_value=1)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated Samples")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def find_latest_ckpt(path):
    if path.endswith(".ckpt"):
        return path
    ckpt_files = glob.glob(os.path.join(path, "**", "*.ckpt"), recursive=True)
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found under path: {path}")
    ckpt_files.sort(key=os.path.getmtime, reverse=True)
    return ckpt_files[0]


@torch.no_grad()
def generate_dcgan_samples(model_path, num_samples=64, output="dcgan_samples.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DCGAN.load_from_checkpoint(model_path).to(device).eval()
    z_dim = model.hparams.z_dim if hasattr(model.hparams, 'z_dim') else 100
    z = torch.randn(num_samples, z_dim, 1, 1, device=device)

    fake_imgs = model(z).cpu()
    save_image_grid(fake_imgs, output)


@torch.no_grad()
def generate_cgan_samples(model_path, samples_per_class=8, output_dir="cgan_outputs"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CGAN.load_from_checkpoint(model_path).to(device).eval()
    z_dim = model.hparams.latent_dim
    num_classes = model.hparams.n_classes

    os.makedirs(output_dir, exist_ok=True)

    for class_idx in range(num_classes):
        z = torch.randn(samples_per_class, z_dim, device=device)
        labels = torch.full((samples_per_class,), class_idx, dtype=torch.long, device=device)
        fake_imgs = model(z, labels).cpu()

        filename = os.path.join(output_dir, f"class_{class_idx}.png")
        save_image_grid(fake_imgs, filename, nrow=samples_per_class)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["dcgan", "cgan"], required=True, help="Model type: dcgan or cgan")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .ckpt file or logs/ directory")
    parser.add_argument("--out", type=str, default="generated.png", help="Output file or folder name")
    args = parser.parse_args()

    # Automatically resolve latest checkpoint path
    resolved_ckpt = find_latest_ckpt(args.ckpt)

    print(f"Using checkpoint: {resolved_ckpt}")

    if args.type == "dcgan":
        generate_dcgan_samples(resolved_ckpt, output=args.out)
    else:
        generate_cgan_samples(resolved_ckpt, output_dir=args.out)
