# ClassificationFruitVegQuality


## ABOUT
We developed an image classification model based on MobileNetV2. We chose this particular architecture because we would like our project to be used in real-world applications. It's lightweight and gives satisfying results.  
We compared it to SqueezeNet and EfficientNet.  
EfficientNet gave the best metrics results, but its weights were much heavier (they are not included in this repo).   
SqueezeNet was comparable in size but it performed worse, so MobileNet was chosen.

## Requirements
- Python 3
- Poetry

## Installation
1. Install Poetry by following the instructions on their [official website](https://python-poetry.org/docs/#installation).
2. Clone this repository:

   ```bash
   git clone https://github.com/szymonszczurowski/classification-of-vegetable-and-fruit-quality.git
   cd classification-of-vegetable-and-fruit-quality
   ```

3. Have virtual environment inside your project (can be skipped)
   ```bash
   poetry config virtualenvs.in-project
   ```

4. Install dependencies
   ```bash
   poetry install
   ```

5. Command above might not work but update the torchvision package first fixes the problem.
	```bash
	poetry update torchvision
	```

## Dependencies

These are the dependencies that this project is build on. They are all taken care of by poetry.
```toml
python = "^3.10"
kaggle = "^1.6.14"
numpy = "1.26.4"
torch = {version = "^2.3.1+cu121", source = "pytorch"}
pytorch-lightning = "^2.3.1"
torchvision = "^0.18.1"
tensorboard = "^2.17.0"
matplotlib = "^3.9.1"
albumentations = "^1.4.10"
jupyter = "^1.0.0"
```

## Project structure
```bash
|-- data
|   |-- __init__.py
|   |-- processed
|   |-- raw
|-- lightning_logs
|   |-- FruitVegEffNet_logs20240703-104456
|   |-- FruitVegEffNet_logs20240704-054944
|   |-- FruitVegEffNet_logs20240704-062152
|   |-- FruitVegEffNet_logs20240704-062208
|   |-- FruitVegEffNet_logs20240704-111124
|   |-- FruitVegMobNet_logs20240704-094947
|   |-- FruitVegSqueezeNet_logs20240704-103147
|   |-- version_0
|-- models
|   |-- dummy_model.py
|   |-- fruit_veg_effnet
|   |-- fruit_veg_mobilenet
|   |-- fruit_veg_squeezenet
|-- notebooks
|   |-- dummy_notebook.py
|   |-- example.ipynb
|   |-- mobilenet.png
|-- src
    |-- __init__.py
    |-- __pycache__
    |-- data_module.py
    |-- dataset_scripts
    |-- modeling
```

## Training
```bash
--- src
    |-- data_module.py
    |-- dataset_scripts
    |-- modeling
        |-- __init__.py
        |-- models
        |-- predict.py
        |-- test.py
        |-- train.py
```

train.py is responsible for training the models located in models directory.
Script and dataset are structued in such way that one wanting to train specific model just has to change the file path
to the model located in models provided such python file exists.
How to run the training
```bash
poetry run src/modeling/train.py
```
The model will be neatly saved to models in root directory (NOT in src/). 

### Example
```bash
|-- data
|-- lightning_logs
|-- models
|   |-- dummy_model.py
|   |-- fruit_veg_effnet
|   |   |-- epoch=24-val_loss=0.36.ckpt
```

## Data download and augmentation
Our package provides two scripts one regarding data collection other augmentation.

Scripts are located in 
```bash
|-- src
    |-- __init__.py
    |-- __pycache__
    |-- data_module.py
    |-- dataset_scripts
    |   |-- __init__.py
    |   |-- dataset_augmentation.py
    |   |-- download_dataset.py
    |-- modeling
```

download_dataset.py downloads dataset from kaggle in our case this was [Healthy vs Rotten](https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten/data) dataset.
Dataset is downloaded to data/raw directory.
### To download run 
```bash
poetry run src/dataset_scripts/download_dataset.py
```
dataset_augmentation.py aguments dataset  and shows distribiution of data.
The script copies and augments image located in data/raw to data/processed.
The model is trained on dataset located in data/processed directory.
### To augment data run 
```bash
poetry run src/dataset_scripts/dataset_augmentation.py
```

> [!NOTE]
> The progress bar is not implemented so be aware of the dataset scripts execution




