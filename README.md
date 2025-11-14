# Digit Classification - MNIST (0, 5, 8)

A complete command-line application for digit classification using a curated, imbalanced subset of MNIST. 

The project follows all requirements from the **Buzz Solutions** problem statement including:

- Dataset curation using MNIST from torchvision
- Imbalanced class sampling (0, 5, 8)
- Deterministic splitting (train/val/eval)
- PyTorch Lightning model implementation (no pretrained models)
- CLI built with Typer 
- Unit tests using pytest

---

## Project Structure
```text
digit-classification/
│
├── pyproject.toml
├── README.md
├── sample.png           # Example image for testing prediction
│
├── src/
│   └── digit_classification/
│       ├── __init__.py
│       ├── cli.py           # Typer CLI commands
│       ├── data.py          # Custom MNIST dataset & splits
│       ├── model.py         # PyTorch Lightning model
│       └── evaluation.py    # Evaluation metrics
│
└── tests/
    ├── __init__.py
    ├── test_data.py
    ├── test_model.py
    └── test_evaluation.py
```

---

## Dataset

- **Source**: MNIST training set (60,000 images, 28×28 grayscale)
- **Curated subset**: 5,000 images total (imbalanced)
  - **Class 8**: 3,500 images (70%)
  - **Class 0**: 1,200 images (24%)
  - **Class 5**: 300 images (6%)

### Data Splits
Two-stage stratified splitting:
- **Evaluation**: 20% of 5,000 = 1,000 images (held out first)
- **Validation**: 15% of remaining 4,000 = 600 images
- **Training**: Remainder = 3,400 images
- **Reproducibility**: Fixed seed (default: 11122025) ensures identical splits

---

## Model Architecture

The model is implemented using PyTorch Lightning's `LightningModule` and follows a simple CNN architecture:

### CNN Structure
```
Input: [B, 1, 28, 28]
  ↓
Conv2d(1→16, kernel=3, padding=1) → BatchNorm → ReLU → MaxPool(2)
  ↓  [B, 16, 14, 14]
Conv2d(16→32, kernel=3, padding=1) → BatchNorm → ReLU → MaxPool(2)
  ↓  [B, 32, 7, 7]
Flatten → [B, 1568]
  ↓
Linear(1568 → 3)
  ↓
Output: [B, 3] logits
```

### Key Features
- **No pretrained weights** - trained from scratch
- **Class-weighted loss** to handle imbalance
- **Normalization**: Mean/std from training set only (prevents data leakage)
- **Metrics**: Macro-averaged Accuracy and F1 Score

---

## Design Decisions

### Handling Class Imbalance
- **Class weights**: Inverse frequency weighting in cross-entropy loss
  - Class 5 (6% of data) receives higher weight to prevent being ignored
- **Macro metrics**: Equal importance to all classes regardless of frequency
- **Stratified splits**: Maintains class proportions across train/val/eval sets

### Training Configuration
- **Optimizer**: Adam (lr=1e-3)
- **LR Scheduler**: ReduceLROnPlateau
  - Monitors `val_f1_macro`
  - Reduces LR by 0.5× when validation F1 plateaus (patience=2)
- **Early Stopping**: Stops training if no improvement for 3 epochs
- **Checkpointing**: Saves best model based on `val_f1_macro`
- **Max Epochs**: 20 (CPU only, per requirements)

---

## Quick Start (on Linux)
### 1. Create and activate virtual environment
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Clone the repository
```bash
git clone https://github.com/roy-shudipto/digit-classification.git
cd digit-classification
```

### 3. Install package
**Basic installation:**
```bash
pip install -e .
```

**With development/test dependencies:**
```bash
pip install -e '.[dev,test]'
```

---

## Command Line Interface (CLI)

The CLI provides four commands as specified:

### 1. Download the MNIST dataset
```bash
digit-classification download-data --data-dir ../mnist
```
Downloads MNIST training data to the specified directory.


### 2. Train the model
```bash
digit-classification train --data-dir ../mnist --output-dir ../mnist_checkpoint --seed 11122025
```

**Available arguments:**
- `--data-dir`: Path to MNIST data directory (required)
- `--output-dir`: Path to save checkpoints and logs (required)
- `--seed`: Random seed for reproducibility (default: 11122025)
- `--eval-ratio`: Evaluation set fraction (default: 0.20)
- `--val-ratio`: Validation set fraction (default: 0.15)
- `--epochs`: Maximum training epochs, max 20 (default: 20)
- `--learning-rate`: Learning rate (default: 1e-3)
- `--batch-size`: Training batch size (default: 32)
- `--num-workers`: DataLoader worker processes (default: 2)

**Output:**
- Best checkpoint: `{output-dir}/best.ckpt`
- Training logs: `{output-dir}/lightning_logs/`


### 3. Evaluate the model
```bash
digit-classification evaluate --checkpoint-path ../mnist_checkpoint/best.ckpt --data-dir ../mnist
```

**Arguments:**
- `--checkpoint-path`: Path to model checkpoint (required)
- `--data-dir`: Path to MNIST data (required)
- `--batch-size`: Batch size for evaluation (default: 32)
- `--num-workers`: DataLoader workers (default: 4)

**Example output:**
```
              precision    recall  f1-score   support

           0     0.9958    0.9958    0.9958       240
           5     1.0000    0.9833    0.9916        60
           8     0.9971    0.9986    0.9979       700

    accuracy                         0.9970      1000
   macro avg     0.9977    0.9926    0.9951      1000
weighted avg     0.9970    0.9970    0.9970      1000

Confusion matrix:
tensor([[239,   0,   1],
        [  0,  59,   1],
        [  1,   0, 699]])
```

### 4. Predict a digit in an image
```bash
digit-classification predict --checkpoint-path ../mnist_checkpoint/best.ckpt --input-path ./sample.png
```

**Arguments:**
- `--checkpoint-path`: Path to model checkpoint (required)
- `--input-path`: Path to input image (will be converted to 28×28 grayscale) (required)

**Example output (JSON):**
```json
{
  "probs": {
    "0": 0.008339464664459229,
    "5": 2.3278232674783794e-06,
    "8": 0.9916581511497498
  },
  "prediction": 8
}
```

---

## Running Tests
The specification requires pytest tests for critical functionality.
### Install test dependencies
```bash
pip install -e '.[test]'
```

### Run all tests
```bash
pytest tests/ -v
```

### Run tests WITH coverage (terminal report)
```bash
pytest tests/ --cov=digit_classification
```

---

## Development
### Format code
```bash
black src/ tests/
```
### Run linter
```bash
flake8 src/ tests/
```

---

## Author
**Shudipto Sekhar Roy**  
shudipto67@gmail.com