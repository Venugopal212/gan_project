import torch
from torch import nn
import pytorch_lightning as pl
from models.cgan_generator import CGANGenerator
from models.cgan_discriminator import CGANDiscriminator

class CGAN(pl.LightningModule):
    def __init__(self, latent_dim=100, n_classes=9, img_shape=(3, 28, 28), lr=2e-4, b1=0.5, b2=0.999):
        super().__init__()
        self.save_hyperparameters()

        self.generator = CGANGenerator(
            latent_dim=self.hparams.latent_dim,
            n_classes=self.hparams.n_classes,
            img_shape=self.hparams.img_shape,
        )
        self.discriminator = CGANDiscriminator(
            n_classes=self.hparams.n_classes,
            img_shape=self.hparams.img_shape,
        )
        self.adversarial_loss = nn.BCELoss()

        # 🔧 Manual optimization mode
        self.automatic_optimization = False

    def forward(self, z, labels):
        return self.generator(z, labels)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        batch_size = imgs.size(0)
        device = self.device

        # Sample noise
        z = torch.randn(batch_size, self.hparams.latent_dim, device=device)

        # Adversarial ground truths
        valid = torch.ones((batch_size, 1), device=device)
        fake = torch.zeros((batch_size, 1), device=device)

        # Get optimizers
        opt_g, opt_d = self.optimizers()

        # ======================
        #  Train Generator
        # ======================
        gen_imgs = self(z, labels)
        validity = self.discriminator(gen_imgs, labels)
        g_loss = self.adversarial_loss(validity, valid)

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()
        self.log("g_loss", g_loss, prog_bar=True)

        # ======================
        #  Train Discriminator
        # ======================
        real_validity = self.discriminator(imgs, labels)
        d_real_loss = self.adversarial_loss(real_validity, valid)

        fake_validity = self.discriminator(gen_imgs.detach(), labels)
        d_fake_loss = self.adversarial_loss(fake_validity, fake)

        d_loss = 0.5 * (d_real_loss + d_fake_loss)

        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
        self.log("d_loss", d_loss, prog_bar=True)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d]
