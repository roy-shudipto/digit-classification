import os
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
    A customized MNIST dataset that keeps only specified classes with exact sample counts
    and provides utilities for stratified splitting, normalization, and class-weight computation.

    This class wraps torchvision's `MNIST` dataset and applies:

    - Filtering to retain only user-specified classes with fixed sample counts.
    - Remapping from original MNIST digit labels to contiguous class indices.
    - Train-only normalization (mean/std) computation.
    - Two-stage stratified splitting into train/validation/eval.
    - Class weight computation for imbalanced learning.
    """

    def __init__(
        self,
        root: str,
        mnist_class_distribution: dict[int, int],
        seed: int,
        train: bool = True,
    ) -> None:
        """
        Initialize the CustomMNIST.

        Args:
            root (str): Root directory where MNIST will be downloaded or loaded from.
            mnist_class_distribution (dict[int, int]):
                A mapping from original digit labels (0–9) to the number of samples
                to keep for each class. Example: ``{0: 2000, 3: 1500}``.
            seed (int): Random seed used for reproducible sampling and data splits.
            train (bool, optional): Whether to load the MNIST training split.
                Defaults to True.

        Raises:
            ValueError: If the class distribution dictionary is invalid or if
                requested sample counts exceed available MNIST samples.

        """
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

    @property
    def raw_folder(self) -> str:
        """
        Override to use 'MNIST' folder instead of 'CustomMNIST'.

        Returns:
            str: The path to the raw MNIST data directory, resolved as `<root>/MNIST/raw`.
            This ensures consistency with torchvision's standard MNIST layout and prevents
            folder names derived from the subclass (e.g., "CustomMNIST").
        """
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self) -> str:
        """
        Override to use 'MNIST' folder instead of 'CustomMNIST'.

        Returns:
            str: The path to the processed MNIST data directory, resolved as `<root>/MNIST/processed`.
            This ensures consistency with torchvision's standard MNIST layout and prevents
            folder names derived from the subclass (e.g., "CustomMNIST").
        """
        return os.path.join(self.root, "MNIST", "processed")

    @staticmethod
    def _validate_class_distribution(class_dist: dict[int, int]) -> None:
        """
        Validate the user-provided MNIST class distribution.

        Ensures that:
        - Keys are integers in [0, 9].
        - Values are positive integers.
        - Dictionary is non-empty.

        Args:
            class_dist (dict[int, int]): Mapping from MNIST class labels to
                requested sample counts.

        Raises:
            ValueError: If keys or values are invalid, or if the dictionary is empty.
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
        """
        Compute train-only normalization statistics and apply a consistent
        transform to all dataset splits.

        Args:
            indices (list[int]): Indices belonging to the training subset.

        Returns:
            None
        """
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
        Compute inverse-frequency class weights using the labels contained
        in the given subset of indices.

        weight[c] = total_samples / samples_in_class_c

        Args:
            indices (np.ndarray): Array of dataset indices (e.g., train + val)
                used to compute class frequency statistics.

        Returns:
            None

        Raises:
            ValueError: If any class receives zero samples in the provided index set.
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
        Perform a two-stage stratified dataset split:

        Stage 1:
            Split off an evaluation subset of size `eval_ratio`.

        Stage 2:
            From the remaining data, split off a validation subset using
            `val_ratio` (fraction of the remaining data).

        Both stages preserve class distributions (stratification).

        Args:
            eval_ratio (float): Fraction of total samples reserved for evaluation.
                Must be in (0, 1).
            val_ratio (float): Fraction of remaining samples reserved for validation.
                Must be in (0, 1).

        Returns:
            None

        Raises:
            ValueError: If ratios are invalid or leave no samples for training.
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
    Provides a deterministic mapping between original MNIST digit labels
    and new contiguous class indices.

    Example:
        label_counts = {0: 2000, 7: 1500, 9: 1200}
        -> label_to_index  = {0: 0, 7: 1, 9: 2}
        -> index_to_label  = {0: 0, 1: 7, 2: 9}
    """

    def __init__(self, label_counts: dict[int, int]) -> None:
        """
        Initialize the ClassMapper.

        Args:
            label_counts (dict[int, int]): Dictionary whose keys define the set
                of original labels to map.

        Raises:
            ValueError: If the dictionary is empty.
        """
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
