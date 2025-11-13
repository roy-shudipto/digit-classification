import json
import torch
import torchvision.transforms as T
import typer
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets

from .data import CustomMNIST
from .evaluation import eval_classifier
from .model import DigitClassifier

# Create the CLI app
app = typer.Typer(help="Digit classification CLI")

MNIST_CLASS_DISTRIBUTION = {8: 3500, 0: 1200, 5: 300}


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def ensure_file(path_str: str, kind: str) -> Path:
    """
    Validate that a file path exists and is a regular file.

    Args:
        path_str (str): Path provided by the user.
        kind (str): Human-readable descriptor used in error messages.

    Returns:
        Path: A validated and resolved Path object.

    Raises:
        typer.Exit: If the file does not exist or is not a file.
    """
    path = Path(path_str)

    if not path.exists():
        typer.secho(f"Error: {kind} not found at '{path}'.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if not path.is_file():
        typer.secho(f"Error: {kind} path is not a file: '{path}'.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    return path


# ---------------------------------------------------------
# Download Command
# ---------------------------------------------------------
@app.command()
def download_data(
    data_dir: str = typer.Option(
        ..., "--data-dir", help="Directory where MNIST will be downloaded."
    )
) -> None:
    """
    Download the MNIST dataset.

    Args:
        data_dir (str): Directory where MNIST will be stored.

    This downloads only the training split (Lightning/PyTorch automatically
    downloads the test set as needed).
    """
    datasets.MNIST(root=data_dir, train=True, download=True)
    typer.echo(f"MNIST dataset successfully downloaded to: {data_dir}")


# ---------------------------------------------------------
# Training Command
# ---------------------------------------------------------
@app.command()
def train(
    data_dir: str = typer.Option(
        ..., "--data-dir", help="Directory containing MNIST data."
    ),
    output_dir: str = typer.Option(
        ..., "--output-dir", help="Directory to store checkpoints and logs."
    ),
    seed: int = typer.Option(
        11122025, "--seed", help="Random seed for dataset sampling."
    ),
    eval_ratio: float = typer.Option(
        0.20, "--eval_ratio", help="Evaluation set ratio (from full dataset)."
    ),
    val_ratio: float = typer.Option(
        0.15, "--val_ratio", help="Validation split ratio (after eval split)."
    ),
    epochs: int = typer.Option(
        20, "--epochs", min=1, max=20, help="Max training epochs."
    ),
    learning_rate: float = typer.Option(
        1e-3, "--learning_rate", help="Learning rate for optimization."
    ),
    batch_size: int = typer.Option(32, "--batch-size", min=4, help="Batch size."),
    num_workers: int = typer.Option(
        4, "--num_workers", min=0, help="Number of DataLoader worker processes."
    ),
) -> None:
    """
    Train a convolutional digit classifier on a custom MNIST subset.

    This performs:
    - Dataset sampling based on user-defined digit counts
    - Train/val split
    - Lightning training loop with early stopping and checkpointing
    """
    # Ensure reproducibility
    seed_everything(seed, workers=True)

    # Ensure output directory exists
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset preparation
    custom_mnist = CustomMNIST(
        root=data_dir,
        mnist_class_distribution=MNIST_CLASS_DISTRIBUTION,
        seed=seed,
    )
    custom_mnist.apply_two_stage_split(eval_ratio=eval_ratio, val_ratio=val_ratio)

    # Initialize model
    model = DigitClassifier(
        num_classes=custom_mnist.num_classes,
        lr=learning_rate,
        mnist_class_distribution=MNIST_CLASS_DISTRIBUTION,
        index_to_label=custom_mnist.index_to_label,
        seed=seed,
        eval_ratio=eval_ratio,
        val_ratio=val_ratio,
        mean=custom_mnist.train_mean,
        std=custom_mnist.train_std,
        class_weights=custom_mnist.class_weights_tensor,
    )

    # Configure checkpoints + early stopping
    checkpoint_callback = ModelCheckpoint(
        dirpath=out_dir,
        save_top_k=1,
        monitor="val_f1_macro",
        mode="max",
        filename="best",
    )

    early_stopping = EarlyStopping(
        monitor="val_f1_macro",
        mode="max",
        patience=3,
    )

    # Trainer setup
    trainer = Trainer(
        default_root_dir=out_dir,
        accelerator="cpu",
        deterministic=True,
        max_epochs=min(epochs, 20),
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=10,
    )

    # DataLoaders
    train_loader = DataLoader(
        custom_mnist.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        custom_mnist.val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Training loop
    trainer.fit(model, train_loader, val_loader)

    typer.echo(
        f"Training completed. Best checkpoint saved at: {checkpoint_callback.best_model_path}"
    )


# ---------------------------------------------------------
# Evaluate Command
# ---------------------------------------------------------
@app.command()
def evaluate(
    checkpoint_path: str = typer.Option(
        ..., "--checkpoint-path", help="Path to model checkpoint."
    ),
    data_dir: str = typer.Option(..., "--data-dir", help="Directory containing MNIST."),
    batch_size: int = typer.Option(32, "--batch-size", min=4),
    num_workers: int = typer.Option(4, "--num_workers", min=0),
) -> None:
    """
    Evaluate a trained model on the validation set.

    Loads the model, reconstructs the dataset split using the same seed/ratios
    as during training, computes metrics, and prints a classification report.
    """
    # Check checkpoint file
    ckpt_path = ensure_file(checkpoint_path, "checkpoint file")

    # Load Lightning checkpoint safely
    try:
        model = DigitClassifier.load_from_checkpoint(str(ckpt_path), class_weights=None)
        model.eval()
    except Exception as e:
        typer.secho(
            f"Error: failed to restore model from checkpoint: {e}", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    # Rebuild the dataset exactly like during training
    custom_mnist = CustomMNIST(
        root=data_dir,
        mnist_class_distribution=model.mnist_class_distribution,
        seed=model.seed,
    )
    custom_mnist.apply_two_stage_split(
        eval_ratio=model.eval_ratio, val_ratio=model.val_ratio
    )

    eval_loader = DataLoader(
        custom_mnist.val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Evaluate model
    report, cm = eval_classifier(
        model, eval_loader, index_to_label=model.index_to_label
    )

    typer.echo(report)
    typer.echo(f"Confusion matrix:\n{cm}")


# ---------------------------------------------------------
# Predict Command
# ---------------------------------------------------------
@app.command()
def predict(
    checkpoint_path: str = typer.Option(
        ..., "--checkpoint-path", help="Path to the trained model checkpoint."
    ),
    input_path: str = typer.Option(
        ..., "--input-path", help="Path to an input image file."
    ),
) -> None:
    """
    Predict the digit in a single input image.

    Steps:
    - Validate checkpoint + image paths
    - Restore the trained model from checkpoint
    - Apply the same preprocessing as during training
    - Run a forward pass
    - Output class probabilities + predicted label
    """
    # Validate both paths
    ckpt_path = ensure_file(checkpoint_path, "checkpoint file")
    img_path = ensure_file(input_path, "input image")

    # Load trained model
    try:
        model = DigitClassifier.load_from_checkpoint(str(ckpt_path), class_weights=None)
        model.eval()
    except Exception as e:
        typer.secho(
            f"Error: failed to restore model from checkpoint: {e}", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    # Preprocessing identical to training
    transform = T.Compose(
        [
            T.Grayscale(),  # ensure 1 channel
            T.Resize((28, 28)),  # match training resolution
            T.ToTensor(),  # normalize to [0,1]
            T.Normalize((model.mean,), (model.std,)),
        ]
    )

    # Load image safely
    try:
        img = Image.open(img_path)
    except Exception as e:
        typer.secho(f"Error: unable to open input image: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Preprocess
    x = transform(img).unsqueeze(0)  # [1, 1, 28, 28]

    # Prediction
    with torch.no_grad():
        logits = model(x)
        probs_tensor = torch.softmax(logits, dim=1).squeeze(0)

    probs = probs_tensor.tolist()
    pred_idx = int(probs_tensor.argmax().item())

    # Map internal indices to original digit labels
    probs_dict = {str(model.index_to_label[i]): probs[i] for i in range(len(probs))}

    output = {
        "probs": probs_dict,
        "prediction": model.index_to_label[pred_idx],
    }

    typer.echo(json.dumps(output, indent=2))


if __name__ == "__main__":
    app()
