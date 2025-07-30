import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from gan.cgan_module import CGAN
from data.pathmnist_loader import get_pathmnist_loaders
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Load data
    train_loader, _ = get_pathmnist_loaders(batch_size=128, root="./data/medmnist")

    # Initialize model
    model = CGAN()

    # Setup logger
    logger = TensorBoardLogger("logs", name="cgan")

    # Trainer with logger
    trainer = Trainer(
        max_epochs=100,
        accelerator="cpu",
        devices=1,
        logger=logger,
        log_every_n_steps=50
    )

    # Train model
    trainer.fit(model, train_loader)
