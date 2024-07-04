import os
import sys

from pytorch_lightning import Trainer

from models.FruitVegMobileNet import FruitVegMobNet
from src.data_module import FruitVegDatamodule


# This code is needed to run the script as for now the imports are not working otherwise
# sys.path.insert(
#     0, os.getcwd()
# )
# from models.FruitVegMobileNet import FruitVegMobNet
# from src.data_module import FruitVegDatamodule
# TODO  add the test
def test_model(model_path: str) -> None:
    """
    Test the trained model on the test dataset.

    Args:
        model_path (str): The path to the trained model checkpoint.

    Returns
    -------
        None

    """
    datamodule = FruitVegDatamodule(batch_size=32)
    datamodule.setup()
    num_classes = len(datamodule.test_dataset.dataset.classes)

    model = FruitVegMobNet.load_from_checkpoint(model_path, num_classes=num_classes)

    trainer = Trainer(accelerator="gpu", logger=False)

    results = trainer.test(model=model, datamodule=datamodule)
    print(results)


if __name__ == "__main__":
    model_path = "models/fruit_veg_mobilenet/epoch=24-val_loss=0.23.ckpt"
    test_model(model_path)
