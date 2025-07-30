import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from models.generator import Generator
from models.discriminator import Discriminator

class GAN2D(pl.LightningModule):
    def __init__(self, z_dim=2, lr_g=1e-4, lr_d=2e-4):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(z_dim=z_dim)
        self.discriminator = Discriminator()
        self.validation_z = torch.randn(2048, z_dim)
        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)

    def generator_step(self, batch):
        real = batch[0] if isinstance(batch, (list, tuple)) else batch
        z = torch.randn(real.size(0), self.hparams.z_dim).type_as(real)
        fake = self(z)
        pred_fake = self.discriminator(fake)
        g_loss = F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake))
        return g_loss

    def discriminator_step(self, batch):
        real = batch[0] if isinstance(batch, (list, tuple)) else batch
        z = torch.randn(real.size(0), self.hparams.z_dim).type_as(real)
        fake = self(z).detach()
        pred_real = self.discriminator(real)
        pred_fake = self.discriminator(fake)
        real_loss = F.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real))
        fake_loss = F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake))
        d_loss = (real_loss + fake_loss) / 2
        return d_loss

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()

        # Discriminator step
        opt_d.zero_grad()
        d_loss = self.discriminator_step(batch)
        self.manual_backward(d_loss)
        opt_d.step()

        # Generator step
        opt_g.zero_grad()
        g_loss = self.generator_step(batch)
        self.manual_backward(g_loss)
        opt_g.step()

        self.log("d_loss", d_loss, prog_bar=True)
        self.log("g_loss", g_loss, prog_bar=True)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr_g, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr_d, betas=(0.5, 0.999))
        return [opt_g, opt_d]
