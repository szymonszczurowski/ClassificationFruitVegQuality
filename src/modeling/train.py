import os
import sys
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.data_module import FruitVegDatamodule
from src.modeling.models.FruitVegEffNet import FruitVegEffNet

# This code is needed to run the script as for now the imports are not working otherwise
# sys.path.insert(
#     0, os.getcwd()
# )
# from src.modeling.models.FruitVegEffNet import FruitVegEffNet
# from src.data_module import FruitVegDatamodule


if __name__ == "__main__":
    dm = FruitVegDatamodule()
    dm.setup()
    num_classes = len(dm.train_dataset.dataset.classes)
    print(num_classes)
    print(torch.cuda.is_available())

    checkpoint_callback = ModelCheckpoint(
        dirpath="models/fruit_veg_effnet",  # Directory to save the models
        filename="{epoch}-{val_loss:.2f}",  # Filename format
        save_top_k=1,  # Save the top k models according to the monitored metric
        monitor="val_loss",  # Metric to monitor for saving the best model
        mode="min",  # Mode to monitor (minimize the monitored metric)
        save_weights_only=True,  # If True, only the model's weights will be saved
    )

    model = FruitVegEffNet(num_classes=num_classes)
    logger_file_name = (
        model.__class__.__name__ + "_logs" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = TensorBoardLogger("lightning_logs", name=logger_file_name)
    trainer = pl.Trainer(
        accelerator="cuda",
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=15,
        log_every_n_steps=1,
    )
    trainer.fit(model, dm)
