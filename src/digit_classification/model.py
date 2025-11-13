import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from typing import Tuple, Union

from .data import ClassMapper


class DigitClassifier(LightningModule):
    """
    A convolutional neural network for multiclass digit classification using PyTorch Lightning.

    This model consists of:
    - Two convolutional blocks with batch normalization, ReLU, and max pooling.
    - A fully connected layer for classification.
    - Optional class weighting support for imbalanced datasets.
    - Macro accuracy and macro F1-score metrics on the validation set.
    """

    def __init__(
        self,
        num_classes: int,
        lr: float,
        class_mapper: ClassMapper,
        seed: int,
        eval_ratio: float,
        val_ratio: float,
        mean: float,
        std: float,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        """
        Initialize a DigitClassifier instance.

        Args:
            num_classes (int): Number of output classes after remapping labels into contiguous
            indices [0, num_classes-1]. Determined by the CustomMNIST subset.

            lr (float): Learning rate for the optimizer.

            class_mapper (ClassMapper): A bidirectional mapping between:
                - original MNIST digit labels (e.g., {0, 5, 8})
                - internal training indices used by the model.

            seed (int): Random seed used to deterministically subsample MNIST.

            eval_ratio (float): Fraction of the full sampled dataset reserved for the evaluation split.

            val_ratio (float): Fraction of the evaluation split reserved for the validation subset.

            class_weights (torch.Tensor | None): Optional 1D tensor of shape [num_classes] containing per-class
            weighting coefficients for cross-entropy loss.
        """
        super().__init__()

        self.num_classes = num_classes
        self.lr = lr
        self.class_mapper = class_mapper
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

    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
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
