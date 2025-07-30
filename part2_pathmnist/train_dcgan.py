import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from gan.dcgan_module import DCGAN
from data.pathmnist_loader import get_pathmnist_loaders
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    train_loader, _ = get_pathmnist_loaders(batch_size=128, root="./data/medmnist")
    model = DCGAN()

    # Use TensorBoard logger
    logger = TensorBoardLogger("logs", name="dcgan")

    trainer = Trainer(
        max_epochs=100,
        accelerator="cpu",  # or "gpu" if available
        devices=1,
        logger=logger
    )

    trainer.fit(model, train_loader)
