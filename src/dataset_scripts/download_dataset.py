"""
Zaimplementować skrypt pobierający zbiór danych z kaggle'a. Skrypt powinien być w stanie pobrać zbiór danych z kaggle'a, rozpakować go i zapisać w data/raw.

To tylko przykładowy skrypt, nadal musimy go zaimplementować poprawnie.
"""
import os
import shutil
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi

DATA_DIR = "data/raw"


def download_data(dataset_name: str, output_dir: str = DATA_DIR) -> None:
    """
    Download and extract a dataset from Kaggle.

    Args:
    ----
        dataset_name: The Kaggle dataset's name in 'owner/dataset-name' format.
        output_dir: The directory to download and extract the dataset to.

    """
    api = KaggleApi()
    api.authenticate()

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    api.dataset_download_files(dataset_name, path=output_dir, unzip=False)

    # Extract the downloaded zip file
    with zipfile.ZipFile(
        os.path.join(output_dir, f"{dataset_name.split('/')[1]}.zip"), "r"
    ) as zip_ref:
        zip_ref.extractall(output_dir)

    # Remove the downloaded zip file
    os.remove(os.path.join(output_dir, f"{dataset_name.split('/')[1]}.zip"))


if __name__ == "__main__":
    dataset_name = "muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten"
    output_dir = "data/raw"
    download_data(dataset_name, output_dir)
