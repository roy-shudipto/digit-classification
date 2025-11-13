# Digit Classification - MNIST (0, 5, 8)

A complete command-line application for digit classification using a curated, imbalanced subset of MNIST. 

The project follows all requirements from the **Buzz Solutions** problem statement including:

- Dataset curation using MNIST from torchvision
- Imbalanced class sampling (0, 5, 8)
- Deterministic splitting (train/val/eval)
- PyTorch Lightning model implementation (no pretrained models)
- CLI built with Typer 
- Unit tests using pytest

## Project Structure
```text
digit-classification/
│
├── pyproject.toml
├── README.md
│── sample.png
│
├── src/
│   └── digit_classification/
│       ├── __init__.py
│       ├── cli.py
│       ├── data.py
│       ├── model.py
│       ├── evaluation.py
│
└── tests/
    ├── __init__.py
    ├── test_data.py
    ├── test_model.py
    ├── test_evaluation.py
```

## Model Architecture

The model is implemented using PyTorch Lightning's `LightningModule` and follows a simple CNN backbone:

### CNN Architecture

- `Conv2d → BatchNorm → ReLU → MaxPool`

- `Conv2d → BatchNorm → ReLU → MaxPool`

- `Flatten`

- `Fully connected` classifier

- Output logits over `3 classes`

### Additional Features
- Support for class weights to handle imbalance

- Softmax applied in `predict_step`

- Deterministic behavior using seeded splits

### Metrics

- `Macro Accuracy` (MulticlassAccuracy)

- `Macro F1 Score` (MulticlassF1Score)

## Quick Start

### 1. Clone the repository
```
git clone https://github.com/roy-shudipto/digit-classification.git
cd digit-classification
```
### 2. Create and activate virtual environment
```
python3 -m venv .venv
source .venv/bin/activate
```
### 3. Install package
```
pip install -e .
```
Or for development:
```
pip install -e '.[dev,test]'
```

## Command Line Interface (CLI)

The CLI exposes the four required commands:

### 1. Download the dataset
```bash
digit-classification download-data --data-dir ../mnist
```
This downloads MNIST and creates the curated subset (0, 5, 8).

### 2. Train the model
```bash
digit-classification train --data-dir ../mnist --output-dir ../mnist-checkpoint --seed 11122025
```
This trains the CNN and saves the best checkpoint to `output-dir`.
### 3. Evaluate a trained model
```bash
digit-classification evaluate --checkpoint-path ../mnist-checkpoint/best.ckpt --data-dir ../mnist
```
Outputs `accuracy` and `macro-F1` on the evaluation split.

### 4. Download the dataset
```bash
digit-classification predict --checkpoint-path ../mnist-checkpoint/best.ckpt --input-path ./sample.png
```
Predicts the digit in a user-provided PNG image.

