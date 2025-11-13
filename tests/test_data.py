import copy
import hashlib
import numpy as np
import pytest
import torch
from collections import Counter
from typing import Any, Dict

from digit_classification.data import ClassMapper, CustomMNIST


# =============================================================
# Helpers
# =============================================================
def md5_image(image: np.ndarray | torch.Tensor) -> str:
    """
    Compute a deterministic MD5 checksum from an image array or tensor.

    Args:
        image (np.ndarray | torch.Tensor): Image data as either a NumPy array
            or a PyTorch tensor of any numeric dtype and shape.

    Returns:
        str: The hexadecimal MD5 hash of the image's raw pixel data.
    """
    if isinstance(image, torch.Tensor):
        image = image.numpy()

    arr = image.astype(np.uint8)
    return hashlib.md5(arr.tobytes()).hexdigest()


# =============================================================
# Fixtures
# =============================================================
@pytest.fixture(
    params=[
        {"class_distribution": {5: 20, 0: 10, 7: 30}, "seed": 1000},
        {"class_distribution": {7: 30, 2: 20, 9: 40}, "seed": 3000},
    ]
)
def config(request) -> Dict[str, Any]:
    """
    Parameterized configuration for CustomMNIST tests.

    Each parameter set provides:
      - class_distribution: A mapping from digit labels to the number of samples to include in the dataset.
      - seed: A random seed used to ensure reproducible sampling and splitting behavior.

    Args:
        request: The built-in pytest request fixture used to access the current parameter set.

    Returns:
        Dict[str, Any]: A configuration dictionary for constructing CustomMNIST instances.
    """
    return request.param


@pytest.fixture
def temp_root(tmp_path) -> str:
    """
    Provide a temporary root directory for CustomMNIST downloads and data.

    The directory is created under pytest's tmp_path fixture and is automatically cleaned up at the end of
    the test session.

    Args:
        tmp_path: Pytest-managed temporary path object.

    Returns:
        str: String path to the temporary root directory.
    """
    return str(tmp_path)


@pytest.fixture
def split_ratios() -> Dict[str, float]:
    """
    Shared split ratios for the two-stage train/val/eval split.

    Returns:
        Dict[str, float]: A dictionary with keys:
            - "eval_ratio": Proportion of the full dataset reserved for the evaluation subset.
            - "val_ratio": Proportion of the remaining training pool (after eval extraction) reserved
            for the validation subset.
    """
    return {"eval_ratio": 0.20, "val_ratio": 0.15}


@pytest.fixture
def dataset(
    config: Dict[str, Any], temp_root: str, split_ratios: Dict[str, float]
) -> CustomMNIST:
    """
    Construct a CustomMNIST instance and apply the two-stage split.

    Args:
        config (Dict[str, Any]): Parameters including the class distribution and seed.
        temp_root (str): Temporary directory to store MNIST data.
        split_ratios (Dict[str, float]): Ratios for eval and validation splits.

    Returns:
        CustomMNIST: A dataset instance with train, validation, and evaluation subsets.
    """
    class_distribution = config["class_distribution"]
    seed = config["seed"]

    custom_mnist = CustomMNIST(
        root=temp_root,
        mnist_class_distribution=class_distribution,
        seed=seed,
    )
    custom_mnist.apply_two_stage_split(
        eval_ratio=split_ratios["eval_ratio"],
        val_ratio=split_ratios["val_ratio"],
    )

    return custom_mnist


@pytest.fixture
def dataset_copy(
    config: Dict[str, Any], temp_root: str, split_ratios: Dict[str, float]
) -> CustomMNIST:
    """
    Construct a second CustomMNIST instance with identical configuration.

    Args:
        config (Dict[str, Any]): Parameters including the class distribution and seed.
        temp_root (str): Temporary directory to store MNIST data.
        split_ratios (Dict[str, float]): Ratios for eval and validation splits.

    Returns:
        CustomMNIST: A second dataset instance, independent of dataset but constructed with the
        same configuration and split parameters.
    """
    class_distribution = config["class_distribution"]
    seed = config["seed"]

    custom_mnist = CustomMNIST(
        root=temp_root,
        mnist_class_distribution=class_distribution,
        seed=seed,
    )
    custom_mnist.apply_two_stage_split(
        eval_ratio=split_ratios["eval_ratio"],
        val_ratio=split_ratios["val_ratio"],
    )

    return custom_mnist


# =============================================================
# Class distribution validation tests
# =============================================================
def test_validate_class_distribution_rejects_empty() -> None:
    """
    _validate_class_distribution should reject an empty distribution dict.

    An empty class distribution is considered invalid, and the class-level helper is expected to
    raise ValueError in that case.
    """
    with pytest.raises(ValueError):
        CustomMNIST._validate_class_distribution({})


@pytest.mark.parametrize("eval_ratio", [-0.1, 0.0, 1.0, 1.1])
def test_apply_two_stage_split_rejects_invalid_eval_ratio(
    dataset: CustomMNIST, eval_ratio: float
) -> None:
    """
    apply_two_stage_split should reject invalid eval_ratio values.

    The method is expected to raise a ValueError when the evaluation split ratio is not strictly between 0 and 1.

    Args:
        dataset (CustomMNIST): Prepared dataset fixture.
        eval_ratio (float): Candidate evaluation split ratio to validate.
    """
    with pytest.raises(ValueError):
        dataset.apply_two_stage_split(eval_ratio=eval_ratio, val_ratio=0.5)


@pytest.mark.parametrize("val_ratio", [-0.1, 0.0, 1.0, 1.1])
def test_apply_two_stage_split_rejects_invalid_val_ratio(
    dataset: CustomMNIST, val_ratio: float
) -> None:
    """
    apply_two_stage_split should reject invalid val_ratio values.

    The method is expected to raise a ``ValueError`` when the validation split ratio is not strictly between 0 and 1.

    Args:
        dataset (CustomMNIST): Prepared dataset fixture.
        val_ratio (float): Candidate validation split ratio to validate.
    """
    with pytest.raises(ValueError):
        dataset.apply_two_stage_split(eval_ratio=0.2, val_ratio=val_ratio)


# =============================================================
# Dataset behavior tests
# =============================================================
def test_dataset_reproducible(dataset: CustomMNIST, dataset_copy: CustomMNIST) -> None:
    """
    Ensure two identically configured CustomMNIST instances are reproducible.

    This test checks that:
      - The validation subset indices are identical across both datasets.
      - A specific validation sample (both label and image content) matches exactly,
      as verified by the MD5 hash of the image.

    Args:
        dataset (CustomMNIST): First dataset instance.
        dataset_copy (CustomMNIST): Second dataset instance constructed with the same configuration.
    """
    # Test parameter
    val_index = 3

    # Validation subset indices must match
    assert dataset.val_dataset.indices == dataset_copy.val_dataset.indices

    # Compare a specific validation sample
    img1, lbl1 = dataset.val_dataset[val_index]
    img2, lbl2 = dataset_copy.val_dataset[val_index]

    assert lbl1 == lbl2
    assert md5_image(img1) == md5_image(img2)


def test_two_stage_split_sizes(
    dataset: CustomMNIST,
    split_ratios: Dict[str, float],
    config: Dict[str, Any],
) -> None:
    """
    Verify that the two-stage split yields subsets of the expected sizes.

    This test enforces the following conditions:
      1. The overall dataset size equals the sum of the requested class distribution values.
      2. The train/validation/evaluation split fully covers the dataset.
      3. The eval subset size approximately matches: eval_ratio * total.
      4. The validation subset size approximately matches: val_ratio * (total - eval_size).

    Args:
        dataset (CustomMNIST): Dataset instance that has already been split.
        split_ratios (Dict[str, float]): Ratios for eval and validation splits.
        config (Dict[str, Any]): Configuration containing class distribution.
    """
    eval_ratio = split_ratios["eval_ratio"]
    val_ratio = split_ratios["val_ratio"]

    # Dataset size should match requested class distribution
    expected_total = sum(config["class_distribution"].values())
    dataset_size = len(dataset)

    assert dataset_size == expected_total, (
        f"Dataset size ({dataset_size}) does not match "
        f"requested total ({expected_total})"
    )

    # Split sizes must cover the whole dataset
    train_size = len(dataset.train_dataset)
    val_size = len(dataset.val_dataset)
    eval_size = len(dataset.eval_dataset)

    assert train_size + val_size + eval_size == dataset_size

    # Eval size is approx eval_ratio * total
    assert eval_size == pytest.approx(dataset_size * eval_ratio, rel=0.2)

    # Val size is approx val_ratio * remaining after eval
    assert val_size == pytest.approx((dataset_size - eval_size) * val_ratio, rel=0.2)


def test_dataset_class_distribution_matches_config(
    dataset: CustomMNIST, config: Dict[str, Any]
) -> None:
    """
    Check that the realized class distribution matches the requested config.

    Args:
        dataset (CustomMNIST): The constructed dataset instance.
        config (Dict[str, Any]): Configuration containing the requested class distribution.
    """
    # Original requested distribution (digit -> count)
    given_dist = config["class_distribution"]

    # Convert requested digits to mapped class indices
    expected_dist = {
        dataset.label_to_index[digit]: count for digit, count in given_dist.items()
    }

    # Actual distribution in the dataset after mapping
    actual_dist = Counter(int(lbl) for lbl in dataset.targets)

    # Compare sorted versions (order-independent, easier to debug)
    assert dict(sorted(actual_dist.items())) == dict(sorted(expected_dist.items()))


def test_calculate_class_weights_raises_if_class_missing(
    dataset_copy: CustomMNIST, split_ratios: Dict[str, float]
) -> None:
    """
    Ensure class weight calculation fails when a class has zero samples.

    Args:
        dataset_copy (CustomMNIST): A dataset instance used as the base for the simulated failure case.
        split_ratios (Dict[str, float]): Ratios for eval and validation splits.
    """
    # Work on a deep copy to avoid mutating the fixture
    bad_dataset = copy.deepcopy(dataset_copy)

    # Simulate an extra class with zero samples
    bad_dataset.num_classes += 1

    with pytest.raises(ValueError) as excinfo:
        bad_dataset.apply_two_stage_split(
            eval_ratio=split_ratios["eval_ratio"],
            val_ratio=split_ratios["val_ratio"],
        )

    msg = str(excinfo.value)
    assert "zero samples" in msg


# =============================================================
# ClassMapper tests
# =============================================================
def test_class_mapper_bidirectional() -> None:
    """
    Verify that ClassMapper builds consistent forward and reverse mappings.

    Raises:
        AssertionError: If any of the expected mappings do not hold.
    """
    label_counts = {5: 20, 0: 10, 8: 30}
    mapper = ClassMapper(label_counts)

    # Forward mapping: labels -> indices
    assert mapper.label_to_index == {0: 0, 5: 1, 8: 2}

    # Reverse mapping: indices -> labels
    assert mapper.index_to_label == {0: 0, 1: 5, 2: 8}


def test_class_mapper_rejects_empty_dict() -> None:
    """
    ClassMapper should reject an empty label_counts dictionary.

    An empty mapping provides no information about the available classes and is therefore considered invalid.
    A ValueError is expected.
    """
    with pytest.raises(ValueError):
        ClassMapper({})
