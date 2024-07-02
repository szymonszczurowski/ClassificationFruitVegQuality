import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

import src.data_module as md
import src.modeling.models.FruitVegEffNet as model

# sys.path.insert(
#     0, "C:\\Users\\szymo\\PROGRAMMING\\classification-of-vegetable-and-fruit-quality"
# )
# from src.modeling.models.FruitVegEffNet import FruitVegEffNet
# from src.data_module import FruitVegDatamodule

if __name__ == "__main__":
    dm = md.FruitVegDatamodule()
    dm.setup()
    num_classes = len(dm.train_dataset.dataset.classes)
    print(num_classes)
    print(torch.cuda.is_available())

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath="models/fruit_veg_effnet",
    #     filename="{epoch}-{val_loss:.2f}",
    #     save_top_k=1,
    #     monitor="val_loss",
    # )
    # trainer = pl.Trainer(max_epochs=30, accelerator = "gpu", callbacks=[checkpoint_callback])

    model = model.FruitVegEffNet(num_classes=num_classes)
    # logger = TensorBoardLogger("lightning_logs", name="animation_model")
    trainer = pl.Trainer(
        accelerator="cuda",
        # logger = logger,
        max_epochs=30,
        log_every_n_steps=1,
    )
    trainer.fit(model, dm)
