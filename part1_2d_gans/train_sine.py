import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer
from gan.trainer import GAN2D
from data.sine_wave import generate_sine_wave_data
from utils.plot_utils import plot_2d_scatter
import warnings
warnings.filterwarnings("ignore")


# Generate real 2D sine data
data = generate_sine_wave_data(n_samples=2048)
loader = DataLoader(TensorDataset(data), batch_size=256, shuffle=True)

# Initialize GAN model
model = GAN2D()
trainer = Trainer(max_epochs=2000)
trainer.fit(model, loader)

# After training: generate fake data
model.eval()
with torch.no_grad():
    z = torch.randn(2048, model.hparams.z_dim)
    fake_samples = model(z)

# Plot real vs fake
plot_2d_scatter(real=data, fake=fake_samples, title="GAN: Real vs Fake (Sine Wave)", save_path="sine_gan_output.png")
