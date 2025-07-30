import torch
import pytorch_lightning as pl
import torchvision
import torch.nn.functional as F
from models.dcgan_generator import DCGANGenerator
from models.dcgan_discriminator import DCGANDiscriminator
from utils.eval_metrics import get_evaluation_metrics, preprocess_for_eval


class DCGAN(pl.LightningModule):
    def __init__(self, z_dim=100, lr=2e-4, beta1=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.generator = DCGANGenerator(z_dim=z_dim)
        self.discriminator = DCGANDiscriminator()
        self.fixed_noise = torch.randn(64, z_dim, 1, 1)
        self.automatic_optimization = False  # required for multiple optimizers

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        images, _ = batch
        images = torch.nn.functional.interpolate(images, size=(64, 64), mode="bilinear")
        batch_size = images.size(0)
        device = images.device

        # === Discriminator step ===
        z = torch.randn(batch_size, self.hparams.z_dim, 1, 1).type_as(images)
        fake = self(z).detach()
        real_pred = self.discriminator(images)
        fake_pred = self.discriminator(fake)
        d_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred)) +
            F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
        )
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        # === Generator step ===
        z = torch.randn(batch_size, self.hparams.z_dim, 1, 1).type_as(images)
        fake = self(z)
        pred = self.discriminator(fake)
        g_loss = F.binary_cross_entropy_with_logits(pred, torch.ones_like(pred))
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        self.log("d_loss", d_loss, prog_bar=True)
        self.log("g_loss", g_loss, prog_bar=True)

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        return [g_opt, d_opt]

    def on_train_epoch_end(self):
        # Log generated images
        with torch.no_grad():
            fake = self(self.fixed_noise.type_as(self.generator.net[0].weight)).detach().cpu()
            grid = torchvision.utils.make_grid(fake, normalize=True, nrow=8)
            
            # Check if logger supports image logging
            if hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "add_image"):
                self.logger.experiment.add_image("Generated Samples", grid, self.current_epoch)

            # Evaluation metrics (optional)
            try:
                real_imgs = next(iter(self.trainer.datamodule.train_dataloader()))[0][:100].type_as(fake)
                real_eval = preprocess_for_eval(real_imgs)
                fake_eval = preprocess_for_eval(fake)

                fid, is_score, kid = get_evaluation_metrics(device=self.device)
                fid.update(real_eval, real=True)
                fid.update(fake_eval, real=False)
                kid.update(real_eval, real=True)
                kid.update(fake_eval, real=False)
                is_score.update(fake_eval)

                fid_val = fid.compute().item()
                is_val = is_score.compute()[0].item()
                kid_val = kid.compute()[0].item()

                self.log_dict({
                    "FID": fid_val,
                    "Inception Score": is_val,
                    "KID": kid_val
                }, prog_bar=True)
            except Exception as e:
                print(f"[Warning] Skipped evaluation metrics due to: {e}")
