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
def ensure_file(path_str: str) -> Path:
    """
    Validate that a file path exists and is a regular file.

    Args:
        path_str (str): Path provided by the user.

    Returns:
        Path: A validated and resolved Path object.

    Raises:
        typer.Exit: If the file does not exist or is not a file.
    """
    path = Path(path_str)

    if not path.exists():
        typer.secho(f"Error: {path} is not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if not path.is_file():
        typer.secho(f"Error: {path} is not a file.", fg=typer.colors.RED)
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
    Download the MNIST training dataset into the specified directory.

    Args:
        data_dir (str): Directory where the MNIST data will be stored. If the
            dataset is already present, it will not be downloaded again.

    Returns:
        None.
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
        0.20, "--eval-ratio", help="Evaluation set ratio (from full dataset)."
    ),
    val_ratio: float = typer.Option(
        0.15, "--val-ratio", help="Validation split ratio (after eval split)."
    ),
    epochs: int = typer.Option(
        20,
        "--epochs",
        min=1,
        max=20,
        help="Number of training epochs. Must be between 1 and 20.",
    ),
    learning_rate: float = typer.Option(
        1e-3, "--learning-rate", help="Learning rate for optimization."
    ),
    batch_size: int = typer.Option(32, "--batch-size", min=4, help="Batch size."),
    num_workers: int = typer.Option(
        2, "--num-workers", min=0, help="Number of DataLoader worker processes."
    ),
) -> None:
    """
    Train a convolutional digit classifier on a custom MNIST subset.

    This command:
      - Builds a CustomMNIST dataset with a fixed class distribution.
      - Applies a two-stage split into train/val/eval subsets.
      - Initializes a DigitClassifier with normalization stats and class weights.
      - Runs a Lightning training loop with early stopping and checkpointing.

    Args:
        data_dir (str): Directory containing the MNIST training data.
        output_dir (str): Directory where checkpoints and logs are written.
        seed (int): Random seed used for dataset subsampling and splitting.
        eval_ratio (float): Proportion of the full dataset reserved for evaluation.
        val_ratio (float): Fraction of the remaining data (after eval split)
            to use for validation.
        epochs (int): Maximum number of training epochs (capped at 20).
        learning_rate (float): Learning rate for the Adam optimizer.
        batch_size (int): Batch size for training and validation DataLoaders.
        num_workers (int): Number of worker processes for DataLoaders.

    Returns:
        None.
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
        max_epochs=epochs,
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
    num_workers: int = typer.Option(2, "--num-workers", min=0),
) -> None:
    """
    Evaluate a trained model on the evaluation subset of the custom MNIST dataset.

    The command:
      - Validates and loads a saved checkpoint.
      - Reconstructs the dataset using the same class distribution, seed, and
        split ratios that were used during training.
      - Runs eval_classifier to compute metrics and the confusion matrix.
      - Prints the classification report and confusion matrix to terminal.

    Args:
        checkpoint_path (str): Path to the trained model checkpoint (.ckpt file).
        data_dir (str): Directory containing the MNIST training data.
        batch_size (int): Batch size for the evaluation DataLoader.
        num_workers (int): Number of worker processes for the evaluation DataLoader.

    Returns:
        None.
    """
    # Check checkpoint file
    ckpt_path = ensure_file(checkpoint_path)

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
        custom_mnist.eval_dataset,
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
    Predict the digit in a single input image using a trained model checkpoint.

    The command:
      - Validates the checkpoint and image paths.
      - Restores the trained DigitClassifier from the checkpoint.
      - Applies the same preprocessing pipeline used during training
        (grayscale, resize to 28x28, normalization).
      - Runs a forward pass to obtain class probabilities.
      - Prints a JSON object containing per-class probabilities and the
        predicted digit label.

    Args:
        checkpoint_path (str): Path to the trained model checkpoint (.ckpt file).
        input_path (str): Path to the input image to classify.

    Returns:
        None.
    """
    # Validate both paths
    ckpt_path = ensure_file(checkpoint_path)
    img_path = ensure_file(input_path)

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
    try:
        x = transform(img).unsqueeze(0)
        # Quick sanity check
        if x.shape != torch.Size([1, 1, 28, 28]):
            raise ValueError(f"Expected shape [1, 1, 28, 28], got {x.shape}")
    except Exception as e:
        typer.secho(f"Error: failed to preprocess image: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

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
