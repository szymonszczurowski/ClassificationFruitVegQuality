import pytorch_lightning as pl

from src.data_module import FruitVegDatamodule
from src.modeling.models.FruitVegEffNet import FruitVegEffNet

if __name__ == "__main__":
    num_classes = 131
    dm = FruitVegDatamodule()
    dm.setup()

    checkpoint_callback = pl.ModelCheckpoint(
        dirpath="models/models",
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
    )
    trainer = pl.Trainer(max_epochs=30, gpus=1, callbacks=[checkpoint_callback])

    model = FruitVegEffNet(num_classes=num_classes)
    # logger = TensorBoardLogger("lightning_logs", name="animation_model")
    trainer = pl.Trainer(
        accelerator="gpu",
        # logger = logger,
        max_epochs=30,
        log_every_n_steps=1,
    )
