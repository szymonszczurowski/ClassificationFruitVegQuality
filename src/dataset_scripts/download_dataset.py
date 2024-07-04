import os
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi


def download_dataset(
    kaggle_creds_path: str, dataset_name: str, download_path: str
) -> None:
    """
    Download a dataset from Kaggle using the Kaggle API.

    Args:
        kaggle_creds_path (str): The path to the Kaggle credentials file.
        dataset_name (str): The name of the dataset on Kaggle.
        download_path (str): The path where the dataset should be downloaded.

    Returns
    -------
        None

    """
    api = KaggleApi()
    api.authenticate()
    os.environ["KAGGLE_CONFIG_DIR"] = os.path.abspath(
        os.path.dirname(kaggle_creds_path)
    )

    api.dataset_download_files(dataset_name, path=download_path, unzip=False)


def extract_dataset(zip_path: str, extract_path: str) -> None:
    """
    Extract a dataset from a zip file to the specified extract path.

    Args:
        zip_path (str): The path to the zip file.
        extract_path (str): The path where the dataset will be extracted.

    Returns
    -------
        None

    """
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    os.remove(zip_path)


if __name__ == "__main__":
    kaggle_creds_path = "kaggle.json"

    dataset_name = "muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten"
    download_path = "data/raw"
    extract_path = download_path
    zip_path = "data/raw/fruit-and-vegetable-disease-healthy-vs-rotten.zip"

    download_dataset(kaggle_creds_path, dataset_name, download_path)
    extract_dataset(zip_path, extract_path)
