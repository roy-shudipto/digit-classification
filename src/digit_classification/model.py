import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from typing import Tuple, Union


class DigitClassifier(LightningModule):
    """
    A PyTorch Lightning module implementing a convolutional neural network for multiclass
    digit classification.

    The architecture includes:
    - Two convolutional blocks, each with batch normalization, ReLU activation, and max pooling.
    - A fully connected output layer for classification.
    - Optional per-class weighting to address class imbalance.
    - Macro-averaged accuracy and F1-score computed during validation.
    """

    def __init__(
        self,
        num_classes: int,
        lr: float,
        mnist_class_distribution: dict,
        index_to_label: dict,
        seed: int,
        eval_ratio: float,
        val_ratio: float,
        mean: float,
        std: float,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        """
        Initialize the DigitClassifier.

        Args:
            num_classes (int): Number of output classes after remapping MNIST labels into a
                contiguous range [0, num_classes - 1].

            lr (float): Learning rate used by the optimizer.

            mnist_class_distribution (dict): Dictionary describing the class frequencies in the
                custom MNIST subset.

            index_to_label (dict): Mapping from internal class indices to the original MNIST
                labels (e.g., {0: 0, 1: 5, 2: 8}).

            seed (int): Random seed for deterministic dataset subsampling.

            eval_ratio (float): Proportion of the sampled dataset allocated to the evaluation split.

            val_ratio (float): Fraction of the evaluation split reserved for validation.

            mean (float): Mean used for input normalization.

            std (float): Standard deviation used for input normalization.

            class_weights (torch.Tensor | None): Optional 1D tensor of length `num_classes`
                specifying class weights for the cross-entropy loss to mitigate imbalance.
        """
        super().__init__()

        self.num_classes = num_classes
        self.lr = lr
        self.mnist_class_distribution = mnist_class_distribution
        self.index_to_label = index_to_label
        self.seed = seed
        self.eval_ratio = eval_ratio
        self.val_ratio = val_ratio
        self.mean = mean
        self.std = std
        self.class_weights = class_weights

        # Save hyperparameters except for the (potentially large) weights tensor
        self.save_hyperparameters(ignore=["class_weights"])

        # Convolutional feature extractor
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Fully connected classification head
        self.fc = nn.Linear(32 * 7 * 7, num_classes)

        # Validation metrics
        self.acc = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure and return the optimizer for training.

        Returns:
            torch.optim.Optimizer: The Adam optimizer initialized with the model's learning rate.
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape [B, 1, 28, 28].

        Returns:
            torch.Tensor: Logits of shape [B, num_classes].
        """
        z = self.net(x)
        z = z.view(z.size(0), -1)
        logits = self.fc(z)
        return logits

    def _step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str
    ) -> torch.Tensor:
        """
        Shared logic for training and validation steps.

        Args:
            batch (tuple): The input batch containing (x, y).
            stage (str): Either "train" or "val", used for logging.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        x, y = batch
        logits = self(x)

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss = F.cross_entropy(logits, y, weight=weights)
        else:
            loss = F.cross_entropy(logits, y)

        preds = logits.argmax(dim=1)

        # Only update metrics for validation stage
        if stage == "val":
            self.acc.update(preds, y)
            self.f1.update(preds, y)

        self.log(f"{stage}_loss", loss, prog_bar=True)

        return loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Perform a single training step.

        Args:
            batch (tuple): The input batch containing (x, y).
            batch_idx (int): Index of the current batch within the training epoch.

        Returns:
            torch.Tensor: The computed training loss.
        """
        return self._step(batch, "train")

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Perform a single validation step.

        Args:
            batch (tuple): The input batch containing (x, y).
            batch_idx (int): Index of the current batch within the validation epoch.

        Returns:
            torch.Tensor: The computed validation loss.
        """
        return self._step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        """
        Compute and log validation metrics at the end of each epoch.
        """
        val_acc = self.acc.compute()
        val_f1 = self.f1.compute()

        self.log("val_acc_macro", val_acc, prog_bar=True)
        self.log("val_f1_macro", val_f1, prog_bar=True)

        self.acc.reset()
        self.f1.reset()

    def predict_step(
        self,
        batch: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Generate probability predictions for inference.

        Args:
            batch (tuple or torch.Tensor): The input batch. Can be (x, y) or just x.
            batch_idx (int): Index of the current batch within the prediction loop.

        Returns:
            torch.Tensor: Probability distributions over classes for each sample.
        """
        x, _ = batch if isinstance(batch, (tuple, list)) else (batch, None)
        probs = F.softmax(self(x), dim=1)
        return probs
