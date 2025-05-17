
# Image Classifier Project

This project provides tools to train and use deep learning models for image classification using PyTorch. It includes scripts for training (`train.py`), prediction (`predict.py`), and an interactive Jupyter notebook (`Image Classifier Project.ipynb`).

---

## train.py

**Purpose:**  
Train a deep learning model (VGG16 or ResNet18) on an image dataset using transfer learning.

**Features:**
- Loads and preprocesses image data.
- Supports model selection and hyperparameter tuning.
- Saves trained model checkpoints.
- GPU support for faster training.

**Usage:**
```bash
python train.py <data_directory> --save_dir <save_directory> --arch <model_architecture> --learning_rate <learning_rate> --hidden_units <hidden_units> --epochs <num_epochs> --gpu
```
**Example:**
```bash
python train.py ./flowers/train/ --gpu
```

**Arguments:**
- `argument 1` (required): Path to dataset.
- `--save_dir`: Directory to save checkpoint (default: current).
- `--arch`: Model architecture (`vgg16` or `resnet18`, default: `resnet18`).
- `--learning_rate`: Learning rate (default: 0.001).
- `--hidden_units`: Hidden units in classifier (default: 512).
- `--epochs`: Number of epochs (default: 5).
- `--gpu`: Use GPU if available.

---

## predict.py

**Purpose:**  
Predict image classes using a trained model checkpoint.

**Features:**
- Loads a saved model checkpoint.
- Processes input images for prediction.
- Returns top K predicted classes and probabilities.
- Maps class indices to names with a JSON file.

**Usage:**
```bash
python predict.py <image_path> --checkpoint <checkpoint_path> --top_k <top_k> --category_names <category_names_json> --gpu
```
**Example:**
```bash
python predict.py ./flowers/test/9/image_06410.jpg --gpu
```

**Arguments:**
- `argument 1` (required): Path to input image.
- `--checkpoint` (required): Path to model checkpoint.
- `--top_k`: Number of top predictions (default: 5).
- `--category_names`: JSON file mapping classes to names.
- `--gpu`: Use GPU if available.

---

## Image Classifier Project.ipynb

**Purpose:**  
Interactive notebook for developing, training, evaluating, and visualizing an image classifier.

**Sections:**
1. Data loading and preprocessing
2. Model architecture and transfer learning
3. Training and validation
4. Testing and evaluation
5. Inference and visualization

**How to use:**  
Open the notebook in Jupyter, run cells sequentially, and modify parameters as needed.

---

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- numpy
- matplotlib
- PIL
- Jupyter Notebook (for `.ipynb`)

**Note:**  
Datasets should be organized in `train`, `valid`, and `test` subfolders. Use the `--gpu` flag only if a CUDA-compatible GPU is available.
```
