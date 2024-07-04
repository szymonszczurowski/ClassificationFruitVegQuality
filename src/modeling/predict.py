import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import DataLoader

from models.FruitVegMobileNet import FruitVegMobNet
from src.data_module import FruitVegDatamodule

# This code is needed to run the script as for now the imports are not working otherwise
# sys.path.insert(0, os.getcwd())


def load_data_module(dm: LightningDataModule) -> DataLoader:
    """
    Load the data module and return a DataLoader object.

    Returns
    -------
    DataLoader
        A DataLoader object containing the test dataset.

    """
    # dm = FruitVegDatamodule()
    dm.setup()
    return DataLoader(dm.test_dataset, batch_size=20, shuffle=True)


def load_model(model_path: str, class_names: list) -> tuple:
    """
    Load a pre-trained model for fruit and vegetable quality classification.

    Args
    ----
    model_path : str
        The path to the pre-trained model checkpoint.
    class_names : list
        A list of class names for the classification task.

    Returns
    -------
    tuple
        A tuple containing the loaded model and the device it is loaded on.

    """
    num_classes = len(class_names)
    model = FruitVegMobNet.load_from_checkpoint(model_path, num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device


def predict(
    model: LightningModule, device: torch.device, test_loader: DataLoader
) -> tuple:
    """
    Predicts the labels for a batch of images using the given model.

    Args
    ----
    model : torch.nn.Module
        The trained model to use for prediction.
    device : torch.device
        The device (CPU or GPU) to perform the prediction on.
    test_loader : torch.utils.data.DataLoader
        The data loader containing the test images.

    Returns
    -------
    tuple
        A tuple containing the input images, true labels, and predicted labels.

    """
    images, labels = next(iter(test_loader))
    images = images.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    return images, labels, predicted


def visualize_predictions(
    images: torch.Tensor,
    labels: torch.Tensor,
    predicted: torch.Tensor,
    class_names: list,
) -> None:
    """
    Visualizes the predicted labels for a set of images alongside their true labels.

    Args
    ----
    images : torch.Tensor
        A tensor containing the images to visualize.
    labels : torch.Tensor
        A tensor containing the true labels for the images.
    predicted : torch.Tensor
        A tensor containing the predicted labels for the images.
    class_names : list
        A list of class names corresponding to the labels.

    Returns
    -------
    None

    """
    plt.figure(figsize=(13, 13))
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        img = images[i].cpu() / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        true_label = class_names[labels[i].cpu()]
        predicted_label = class_names[predicted[i].cpu()]
        plt.title(f"True: {true_label}\nPred: {predicted_label}")
        plt.axis("off")
    plt.show()


if __name__ == "__main__":
    model_path = "models/fruit_veg_mobilenet/epoch=24-val_loss=0.33.ckpt"
    dm = FruitVegDatamodule()
    test_loader = load_data_module(dm)
    class_names = test_loader.dataset.dataset.classes
    model, device = load_model(model_path, class_names)
    images, labels, predicted = predict(model, device, test_loader)
    visualize_predictions(images, labels, predicted, class_names)
