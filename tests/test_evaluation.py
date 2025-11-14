import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader

from digit_classification.evaluation import eval_classifier
from digit_classification.model import DigitClassifier


# =============================================================
# Fixtures for testing Evaluation
# =============================================================
@pytest.fixture
def index_to_label() -> dict[int, int]:
    """
    Return a deterministic mapping from internal class indices to original MNIST labels.

    Returns:
        dict[int, int]: Keys are model class indices (0, 1, 2, ...), and values
        are the corresponding MNIST digit labels.
    """
    return {0: 0, 1: 5, 2: 8}


@pytest.fixture
def digit_classifier(index_to_label: dict[int, int]) -> DigitClassifier:
    """
    Instantiate a DigitClassifier.

    Args:
        index_to_label (dict[int, int]): Mapping from model output indices to
            MNIST digit labels.

    Returns:
        DigitClassifier: A classifier instance initialized with random parameters.
    """
    return DigitClassifier(
        num_classes=len(index_to_label),
        lr=1e-3,
        mnist_class_distribution={0: 10, 5: 10, 8: 10},
        index_to_label=index_to_label,
        seed=1000,
        eval_ratio=0.20,
        val_ratio=0.15,
        mean=0.0,
        std=1.0,
        class_weights=None,
    )


@pytest.fixture
def test_loader() -> DataLoader:
    """
    Create a small, deterministic DataLoader containing synthetic MNIST-like data.

    This dataloader provides a fixed set of randomly generated images shaped like
    MNIST digits (1×28×28) along with integer labels in the valid class range.

    Returns:
        DataLoader: A dataloader that yields batches of synthetic image/label pairs.
    """
    num_samples = 6
    x = torch.randn(num_samples, 1, 28, 28)
    y = torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.long)

    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=2, shuffle=False)


# =============================================================
# Test Evaluation
# =============================================================
def test_eval_classifier_with_digit_classifier(
    digit_classifier: DigitClassifier,
    test_loader: DataLoader,
    index_to_label: dict[int, int],
) -> None:
    """
    Integration test ensuring eval_classifier runs end-to-end as expected.

    This test validates several important aspects of the evaluation pipeline:
      - eval_classifier executes without raising any exceptions.
      - The returned classification report is a non-empty string.
      - The confusion matrix has the correct shape (N×N) and dtype (int64).
      - The report includes the original MNIST digit labels (e.g., 0, 5, 8),
        verifying that label translation from internal indices is functioning.

    Args:
        digit_classifier (DigitClassifier): The classifier instance under test.
        test_loader (DataLoader): A dataloader providing synthetic evaluation data.
        index_to_label (dict[int, int]): Mapping from model output indices to
            MNIST digit labels.

    Returns:
        None.

    Raises:
        AssertionError: If any validation condition fails.
    """
    report, cm = eval_classifier(
        model=digit_classifier,
        test_loader=test_loader,
        index_to_label=index_to_label,
    )

    num_classes = len(index_to_label)

    # Check: Classification report
    assert isinstance(report, str)
    assert report.strip(), "Classification report should not be empty."

    # Report text should include the original MNIST labels (e.g., 0, 5, 8)
    for original_label in index_to_label.values():
        assert str(original_label) in report

    # Check: Confusion matrix checks
    assert isinstance(cm, torch.Tensor)
    assert cm.shape == (num_classes, num_classes)
    assert cm.dtype == torch.int64


def test_eval_classifier_confusion_matrix_totals(
    digit_classifier: DigitClassifier,
    test_loader: DataLoader,
    index_to_label: dict[int, int],
) -> None:
    """
    Test that the confusion matrix computed by eval_classifier has consistent totals.

    This verifies three properties:
      - Each row sum equals the number of true samples per class.
      - The total sum of all entries equals the total number of samples.

    Args:
        digit_classifier (DigitClassifier): The trained classifier under test.
        test_loader (DataLoader): DataLoader providing the evaluation dataset.
        index_to_label (dict[int, int]): Mapping of internal class indices to original labels.

    Returns:
        None.
    """
    report, cm = eval_classifier(
        model=digit_classifier,
        test_loader=test_loader,
        index_to_label=index_to_label,
    )

    # Get expected class distribution from test_loader
    all_labels = []
    for _, labels in test_loader:
        all_labels.extend(labels.tolist())

    expected_counts = torch.bincount(
        torch.tensor(all_labels), minlength=len(index_to_label)
    )

    # Row sums should match actual class distribution
    row_sums = cm.sum(dim=1)
    assert torch.equal(
        row_sums, expected_counts
    ), f"Row sums {row_sums} don't match expected counts {expected_counts}"

    # Total sum should equal number of samples
    total_samples = len(all_labels)
    assert (
        cm.sum() == total_samples
    ), f"CM total {cm.sum()} doesn't match sample count {total_samples}"
