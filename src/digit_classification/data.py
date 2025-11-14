import numpy as np
import torch
from collections import Counter
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import MNIST


class CustomMNIST(MNIST):
    """
    A MNIST variant that keeps only specified classes with exact sample counts.
    """

    def __init__(
        self,
        root: str,
        mnist_class_distribution: dict[int, int],
        seed: int,
        train: bool = True,
    ) -> None:
        super().__init__(
            root=root,
            train=train,
            transform=None,
            download=True,
        )
        logger.info(
            f"Original class distribution: {dict(sorted(Counter(self.targets.tolist()).items()))}"
        )

        # Validate user-provided distribution dictionary
        self._validate_class_distribution(mnist_class_distribution)

        # Number of classes
        self.num_classes = len(mnist_class_distribution)

        # Store seed and create an RNG
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Convert labels to NumPy
        mnist_labels = self.targets.numpy()

        # Select indices for each requested class
        selected_indices: list[int] = []
        for class_label, class_size in sorted(mnist_class_distribution.items()):
            class_indices = np.where(mnist_labels == class_label)[0]
            available = class_indices.size
            if class_size > available:
                raise ValueError(
                    f"Requested {class_size} samples for class {class_label}, "
                    f"but only {available} available."
                )

            selected_class_indices = self.rng.choice(
                class_indices, size=class_size, replace=False
            )
            selected_indices.extend(selected_class_indices.tolist())

        # Filter the original MNIST dataset
        self.data = self.data[selected_indices]
        self.targets = self.targets[selected_indices]
        logger.info(
            f"Custom class distribution: {dict(sorted(Counter(self.targets.tolist()).items()))}"
        )

        # Create a consistent class index mapping
        class_mapper: ClassMapper = ClassMapper(mnist_class_distribution)
        logger.info(f"{class_mapper}")
        self.label_to_index: dict = class_mapper.label_to_index
        self.index_to_label: dict = class_mapper.index_to_label

        # Remap labels from raw digits (e.g.: 0->0, 3->1, 8->2)
        remapped = [self.label_to_index[int(lbl)] for lbl in self.targets]
        self.targets = torch.tensor(remapped, dtype=torch.long)
        logger.info(
            f"Custom class distribution after mapping: {dict(sorted(Counter(self.targets.tolist()).items()))}"
        )

        # Placeholders
        self.train_mean = None
        self.train_std = None
        self.train_dataset = None
        self.val_dataset = None
        self.eval_dataset = None
        self.class_weights_tensor = None

    @staticmethod
    def _validate_class_distribution(class_dist: dict[int, int]) -> None:
        """
        Validate the MNIST class distribution dictionary.

        Ensures:
        - Keys are integers between 0 and 9.
        - Values are positive integers.

        Args:
            class_dist: Dictionary mapping class labels to sample counts.

        Raises:
            ValueError: If any class label or count is invalid.
        """
        if not isinstance(class_dist, dict):
            raise ValueError(
                f"Expected dict for class distribution, got {type(class_dist).__name__}"
            )

        if not class_dist:
            raise ValueError("Class distribution dictionary is empty.")

        for class_label, class_count in class_dist.items():
            if not isinstance(class_label, int) or not (0 <= class_label <= 9):
                raise ValueError(
                    f"Invalid MNIST class label {class_label}; expected an int in [0, 9]."
                )

            if not isinstance(class_count, int) or class_count <= 0:
                raise ValueError(
                    f"Invalid count for class {class_label}: must be a positive int (got {class_count})."
                )

    def _apply_transformation(self, indices: list[int]):
        # Compute normalization stats from TRAIN ONLY (uint8 -> float in [0,1])
        self.train_mean = self.data[indices].float().mean().div(255.0)
        self.train_std = self.data[indices].float().std().div(255.0)
        logger.info(
            f"Train-only mean: {self.train_mean:.4f}, std: {self.train_std:.4f}"
        )

        # Set shared transform for all splits
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (self.train_mean.item(),), (self.train_std.item(),)
                ),
            ]
        )

    def _calculate_class_weights(self, indices: np.ndarray) -> None:
        """
        Calculate and store class weights based on the label distribution
        of the given subset of indices (e.g., train + val).

        The weights are computed as inverse frequency:
            weight_c = total_samples / count_c

        Args:
            indices (np.ndarray): Array of indices to consider (e.g., train + val).

        Raises:
            ValueError: If any class has zero samples.
        """
        # Get labels for the selected subset
        targets = np.asarray(self.targets)[indices]

        # Count occurrences for classes that appear
        classes, counts = np.unique(targets, return_counts=True)
        counts = counts.astype(np.float32)

        num_classes = self.num_classes
        weights = np.zeros(num_classes, dtype=np.float32)

        # Total samples across all present classes
        total = counts.sum()

        # Fill in weights for present classes
        for c, count in zip(classes, counts):
            weights[int(c)] = total / count

        # Detect missing classes
        missing = np.where(weights == 0)[0]
        if len(missing) > 0:
            raise ValueError(
                "Class weight calculation failed: the following classes "
                f"have zero samples in the provided indices: {missing.tolist()}"
            )

        self.class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

    def apply_two_stage_split(self, eval_ratio: float, val_ratio: float) -> None:
        """
        Perform a two-stage stratified split of the dataset into train, eval, and validation subsets.

        Stage 1:
            - Splits off a portion of the dataset (defined by `eval_ratio`) for evaluation,
              ensuring that class proportions are preserved.

        Stage 2:
            - From the remaining data, splits off another portion (defined by `val_ratio`)
              for validation, again maintaining class balance.

        Args:
            eval_ratio (float): Fraction of the full dataset to reserve for evaluation.
                                Must be between 0 and 1 (exclusive).
            val_ratio (float): Fraction of the remaining (post-eval) data to reserve for validation.
                               Must be between 0 and 1 (exclusive).

        Returns:
            None
        """
        # Validate ratios
        if not (0 < eval_ratio < 1):
            raise ValueError(f"eval_ratio must be in (0,1), got {eval_ratio}")
        if not (0 < val_ratio < 1):
            raise ValueError(f"val_ratio must be in (0,1), got {val_ratio}")
        if eval_ratio + val_ratio >= 1.0:
            raise ValueError(
                f"eval_ratio ({eval_ratio}) + val_ratio ({val_ratio}) must be < 1.0 "
                f"to leave data for training"
            )

        # Prepare labels/indices
        targets = np.asarray(self.targets)
        all_indices = np.arange(len(targets))

        # Stage 1: hold out eval (stratified on full set)
        remaining_indices, eval_indices = train_test_split(
            all_indices,
            test_size=eval_ratio,
            random_state=self.seed,
            stratify=targets,
        )

        # Stage 2: from remaining, split train/val (stratified on remaining labels)
        remaining_targets = targets[remaining_indices]
        train_indices, val_indices = train_test_split(
            remaining_indices,
            test_size=val_ratio,
            random_state=self.seed,
            stratify=remaining_targets,
        )

        # Apply transformation
        self._apply_transformation(train_indices)

        # Calculate Class Weights
        all_train_val_indices = np.concatenate([train_indices, val_indices])
        self._calculate_class_weights(all_train_val_indices)

        # Create datasets as Subset views of the main dataset
        self.train_dataset = Subset(self, train_indices.tolist())
        self.val_dataset = Subset(self, val_indices.tolist())
        self.eval_dataset = Subset(self, eval_indices.tolist())

        # log dataset counts
        logger.info(
            f"Training distribution after mapping: "
            f"{dict(sorted(Counter(self.targets[train_indices].tolist()).items()))}"
        )
        logger.info(
            f"Validation distribution after mapping: "
            f"{dict(sorted(Counter(self.targets[val_indices].tolist()).items()))}"
        )
        logger.info(
            f"Evaluation distribution after mapping: "
            f"{dict(sorted(Counter(self.targets[eval_indices].tolist()).items()))}"
        )


class ClassMapper:
    """
    Provides a bidirectional mapping between the original dataset labels (e.g., MNIST digits 0, 5, 8)
    and the corresponding model class indices (0, 1, 2).
    """

    def __init__(self, label_counts: dict[int, int]) -> None:
        if not label_counts:
            raise ValueError("label_counts dictionary cannot be empty.")

        # Create contiguous indices based on sorted label order (e.g.: 0->0, 3->1, 8->2)
        self.label_to_index: dict[int, int] = {
            label: idx for idx, label in enumerate(sorted(label_counts.keys()))
        }

        # Reverse lookup: index -> original label (e.g.: 0->0, 1->3, 2->8)
        self.index_to_label: dict[int, int] = {
            idx: label for label, idx in self.label_to_index.items()
        }

    def __repr__(self) -> str:
        return (
            f"ClassMapper: (label_to_index={self.label_to_index}, "
            f"index_to_label={self.index_to_label})"
        )
