import glob
import math
import os
import random
import shutil

import albumentations as A
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


def organize_dataset() -> None:
    """Organize the dataset by copying the raw dataset to the processed dataset path."""
    if not os.path.exists(processed_dataset_path):
        shutil.copytree(raw_dataset_path, processed_dataset_path)


# 2. Analiza danych przed zmianami

# 2.1 Zliczanie klas
def count_classes(path: str) -> list:
    """
    Count the number of classes in the given directory path.

    Args:
        path (str): The directory path containing the classes.

    Returns
    -------
        list: A list of class names.

    """
    dir_list = os.listdir(path)
    print(len(dir_list))
    return dir_list


# 2.2 Zliczanie ilości zdjęć na daną klasę
def count_images(path: str) -> int:
    """
    Count the number of images in a directory.

    Args:
        path (str): The path to the directory containing the images.

    Returns
    -------
        int: The number of images in the directory.

    """
    dir_list = os.listdir(path)
    return len(dir_list)


def analyze_data(classes: list, dataset_path: str) -> dict:
    """
    Analyze the data by counting the number of images in each class.

    Args:
        classes (list): A list of class names.
        dataset_path (str): The path to the dataset directory.

    Returns
    -------
        dict: A dictionary containing the class names as keys and the total number of images as values.

    """
    data = {}
    for element in classes:
        total_images = count_images(os.path.join(dataset_path, element))
        data[element] = total_images

    for key, value in data.items():
        print(key, " : ", value)
    return data


# 2.3 Wykres rozkładu danych
def plot_fruit_and_vegetable_counts(data: dict) -> None:
    """
    Plot the number of images in different categories of fruits and vegetables.

    Args:
        data (dict): A dictionary containing the categories of fruits and vegetables as keys and the number of images as values.

    Returns
    -------
        None

    """
    fruits_and_vegetables = list(data.keys())
    values = list(data.values())

    plt.figure(figsize=(25, 20))
    plt.bar(fruits_and_vegetables, values)

    for i in range(len(fruits_and_vegetables)):
        plt.text(
            i, values[i], values[i], ha="center", bbox=dict(facecolor="red", alpha=0.8)
        )

    plt.xticks(rotation=90)
    plt.xlabel("Fruits And Vegetables", fontsize=18)
    plt.ylabel("No. of images", fontsize=18)
    plt.title("Number Of images in different categories")
    plt.show()


# 3. Redukcja ilości zdjęć
def reduce_images_to_1000(dataset_path: str) -> None:
    """
    Reduce the number of images in each category of a dataset to 1000.

    Args:
        dataset_path (str): The path to the dataset directory.

    Returns
    -------
        None

    """
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if not os.path.isdir(category_path):
            continue

        images = glob.glob(os.path.join(category_path, "*"))
        num_images = len(images)

        if num_images > 1000:
            num_to_remove = num_images - 1000
            print(f"Removing {num_to_remove} images from class {category}")

            images_to_remove = random.sample(images, num_to_remove)
            for image_path in images_to_remove:
                os.remove(image_path)
                print(f"Removed {image_path}")

    print("Completed")


# 4. Augmentacja danych
def augment_images(image_path: str, num_images: int, save_directory: str) -> None:
    """
    Augments the given image by applying a series of transformations and saves the augmented images to a specified directory.

    Args:
        image_path (str): The path to the image file.
        num_images (int): The number of augmented images to generate.
        save_directory (str): The directory to save the augmented images.

    Returns
    -------
        None

    """
    transform = A.Compose(
        [
            A.Rotate(limit=30, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit=0, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ChannelShuffle(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ]
    )

    try:
        image = imageio.imread(image_path)
        if image is None or image.size == 0:
            raise ValueError("Image is empty or corrupted")
    except (OSError, ValueError, AttributeError) as e:
        print(f"Error loading image {image_path}: {e}")
        return

    for i in range(num_images):
        augmented_image = transform(image=image)["image"]
        save_path = os.path.join(
            save_directory, f"aug_{i}_{os.path.basename(image_path)}.png"
        )
        try:
            imageio.imwrite(save_path, augmented_image)
        except (OSError, ValueError) as e:
            print(f"Error saving image {save_path}: {e}")


def augment_dataset(dataset_path: str) -> None:
    """
    Augments the dataset by doubling the number of images for each class that has less than 500 images.

    Args:
    ----
        dataset_path (str): The path to the dataset directory.

    """
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if not os.path.isdir(category_path):
            continue

        images = glob.glob(os.path.join(category_path, "*"))
        num_images = len(images)

        if num_images < 500:
            target_count = num_images * 2
            num_to_augment = target_count - num_images
            print(f"Doubling {num_to_augment} images for class {category}")

            for image_index, image_path in enumerate(images):
                augment_images(image_path, num_to_augment // num_images, category_path)

            remaining_images = target_count - len(
                glob.glob(os.path.join(category_path, "*"))
            )
            for image_path in images[:remaining_images]:
                augment_images(image_path, 1, category_path)

    print("Augmentation completed.")


# 5. Analiza danych po zmianach

# Wyświetlanie jednego zdjęcia z każdej klasy
def show_image(image_path: str, ax: plt.axes, title: str) -> None:
    """
    Display an image on a given axis.

    Args:
    ----------
    image_path (str): The path to the image file.
    ax (matplotlib.axes.Axes): The axis on which to display the image.
    title (str): The title of the image.

    Returns
    -------
    None

    """
    image = imageio.imread(image_path)
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")


def display_sample_images(dataset_path: str) -> None:
    """
    Display sample images from the dataset.

    Args:
        dataset_path (str): The path to the dataset.

    Returns
    -------
        None

    """
    categories = [
        d
        for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ]
    num_categories = len(categories)
    cols = math.ceil(math.sqrt(num_categories))
    rows = math.ceil(num_categories / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))

    for ax, category in zip(axes.flatten(), categories):
        category_path = os.path.join(dataset_path, category)
        image_path = glob.glob(os.path.join(category_path, "*"))[0]
        show_image(image_path, ax, category)

    for ax in axes.flatten()[num_categories:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    raw_dataset_path = "data/raw/Fruit And Vegetable Diseases Dataset"
    processed_dataset_path = "data/processed/Fruit And Vegetable Diseases Dataset"

    organize_dataset()

    classes = count_classes(processed_dataset_path)
    data = analyze_data(classes, processed_dataset_path)
    plot_fruit_and_vegetable_counts(data)

    reduce_images_to_1000(processed_dataset_path)
    augment_dataset(processed_dataset_path)

    classes = count_classes(processed_dataset_path)
    data = analyze_data(classes, processed_dataset_path)
    plot_fruit_and_vegetable_counts(data)
    display_sample_images(processed_dataset_path)
