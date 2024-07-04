import pytorch_lightning as pl
from torch import Generator
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


class FruitVegDatamodule(pl.LightningDataModule):
    """DataModule for the FruitVeg dataset."""

    def __init__(self, batch_size: int = 32) -> None:
        """Initialize the FruitVegDatamodule."""
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: str = None) -> None:
        """Make setup the FruitVeg dataset."""
        transform = transforms.Compose(
            [
                transforms.Resize(200),
                transforms.CenterCrop(200),
                transforms.ToTensor(),
                transforms.Normalize((0), (1)),
            ]
        )

        dataset = ImageFolder(
            root="data/processed/Fruit And Vegetable Diseases Dataset",
            transform=transform,
        )

        seed = Generator().manual_seed(42)
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        self.train_dataset, self.test_dataset, self.val_dataset = random_split(
            dataset, [train_size, test_size, val_size], seed
        )

    def train_dataloader(self) -> DataLoader:
        """Return the DataLoader for the training dataset."""
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8
        )

    def test_dataloader(self) -> DataLoader:
        """Return the DataLoader for the test dataset."""
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8
        )

    def val_dataloader(self) -> DataLoader:
        """Return the DataLoader for the validation dataset."""
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8
        )
