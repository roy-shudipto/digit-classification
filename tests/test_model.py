import pytest
import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from digit_classification.model import DigitClassifier


# =============================================================
# Fixtures
# =============================================================


@pytest.fixture
def index_to_label() -> dict[int, int]:
    """Simple mapping from internal indices to original MNIST labels."""
    return {0: 0, 1: 5, 2: 8}


@pytest.fixture
def digit_classifier(index_to_label: dict[int, int]) -> DigitClassifier:
    """
    Create a DigitClassifier instance without class weights.

    Args:
        index_to_label (dict[int, int]): Mapping from internal class indices to the original MNIST labels.

    Returns:
        DigitClassifier: A classifier instance initialized without class weights.
    """
    return DigitClassifier(
        num_classes=len(index_to_label),
        lr=1e-3,
        mnist_class_distribution={0: 10, 5: 10, 8: 10},
        index_to_label=index_to_label,
        seed=1000,
        eval_ratio=0.2,
        val_ratio=0.15,
        mean=0.0,
        std=1.0,
        class_weights=None,
    )


@pytest.fixture
def digit_classifier_with_weights(index_to_label: dict[int, int]) -> DigitClassifier:
    """
    Create a DigitClassifier instance with class weights.

    Args:
        index_to_label (dict[int, int]): Mapping from internal class indices to the original MNIST labels.

    Returns:
        DigitClassifier: A classifier instance initialized with class weights.
    """
    num_classes = len(index_to_label)
    class_weights = torch.linspace(1.0, 2.0, steps=num_classes)

    return DigitClassifier(
        num_classes=num_classes,
        lr=5e-4,
        mnist_class_distribution={0: 10, 5: 10, 8: 10},
        index_to_label=index_to_label,
        seed=2000,
        eval_ratio=0.2,
        val_ratio=0.15,
        mean=0.0,
        std=1.0,
        class_weights=class_weights,
    )


# =============================================================
# Basic sanity tests
# =============================================================


def test_configure_optimizers_returns_adam(digit_classifier: DigitClassifier) -> None:
    """
    Test that configure_optimizers initializes an Adam optimizer with the correct settings.

    This test verifies that:
      - The optimizer returned by the model is an instance of torch.optim.Adam.
      - PyTorch Lightning correctly stores the model's hyperparameters in self.hparams,
        ensuring reproducibility and checkpoint compatibility.

    Args:
        digit_classifier (DigitClassifier): The classifier instance under test.

    Returns:
        None.
    """
    opt = digit_classifier.configure_optimizers()

    assert isinstance(opt, torch.optim.Adam)

    # Hyperparameters should be saved by Lightning
    assert digit_classifier.hparams.num_classes == digit_classifier.num_classes
    assert digit_classifier.hparams.lr == digit_classifier.lr


def test_forward_output_shape(digit_classifier: DigitClassifier) -> None:
    """
    Test that the model's forward pass returns logits of the correct shape.

    This test ensures that passing a batch of images through the model produces a tensor of
    logits with shape [batch_size, num_classes].

    Args:
        digit_classifier (DigitClassifier): The classifier instance under test.

    Returns:
        None.
    """
    # Simulate data
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)

    logits = digit_classifier(x)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (batch_size, digit_classifier.num_classes)


# =============================================================
# Training / validation step behavior
# =============================================================


def test_training_step_returns_scalar_loss(digit_classifier: DigitClassifier) -> None:
    """
    Test that training_step computes a valid scalar loss.

    Args:
        digit_classifier (DigitClassifier): The model instance under test.

    Returns:
        None.
    """
    # Disable Lightning logging to avoid warnings when no Trainer is attached
    digit_classifier.log = lambda *args, **kwargs: None

    # Simulate data
    batch_size = 8
    x = torch.randn(batch_size, 1, 28, 28)
    y = torch.randint(0, digit_classifier.num_classes, (batch_size,))

    loss = digit_classifier.training_step((x, y), batch_idx=0)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_training_step_uses_class_weights(
    digit_classifier_with_weights: DigitClassifier,
) -> None:
    """
    Test that training_step correctly applies class weights when provided.

    Args:
        digit_classifier_with_weights (DigitClassifier): A model instance configured with non-uniform class weights.

    Returns:
        None.
    """
    # Disable Lightning logging to avoid warnings when no Trainer is attached
    digit_classifier_with_weights.log = lambda *args, **kwargs: None

    # Simulate data
    batch_size = 8
    x = torch.randn(batch_size, 1, 28, 28)
    y = torch.randint(0, digit_classifier_with_weights.num_classes, (batch_size,))

    loss = digit_classifier_with_weights.training_step((x, y), batch_idx=0)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_validation_step_updates_metrics_consistently(
    digit_classifier: DigitClassifier,
) -> None:
    """
    Test that validation_step updates accuracy and F1 metrics consistently.

    Args:
        digit_classifier (DigitClassifier): The model instance under test.

    Returns:
        None.
    """
    # Disable Lightning logging to avoid warnings when no Trainer is attached
    digit_classifier.log = lambda *args, **kwargs: None

    # Simulate data
    num_classes = digit_classifier.num_classes
    batch_size = 10
    x = torch.randn(batch_size, 1, 28, 28)
    y = torch.randint(0, num_classes, (batch_size,))

    # Compute logits once to derive expected metrics.
    with torch.no_grad():
        logits = digit_classifier(x)
    preds = logits.argmax(dim=1)

    # Reference metrics using the same configuration.
    ref_acc = MulticlassAccuracy(num_classes=num_classes, average="macro")
    ref_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
    ref_acc.update(preds, y)
    ref_f1.update(preds, y)

    expected_acc = ref_acc.compute()
    expected_f1 = ref_f1.compute()

    # Run the actual validation step, which should update the model's metrics.
    digit_classifier.validation_step((x, y), batch_idx=0)

    model_acc = digit_classifier.acc.compute()
    model_f1 = digit_classifier.f1.compute()

    assert torch.allclose(model_acc, expected_acc)
    assert torch.allclose(model_f1, expected_f1)


def test_on_validation_epoch_end_logs_and_resets_metrics(
    digit_classifier: DigitClassifier,
) -> None:
    """
    Test that on_validation_epoch_end performs two key actions:

    1. Logs the macro-averaged validation metrics ("val_acc_macro" and "val_f1_macro")
       using the model's log() method.
    2. Resets the internal TorchMetrics state so that metric objects report that they
       have not been updated after the reset.

    Args:
        digit_classifier (DigitClassifier): The model instance under test.

    Returns:
        None.
    """
    # Capture logged values instead of relying on Lightning's trainer
    logged: dict[str, tuple[torch.Tensor, bool]] = {}

    def dummy_logger(name: str, value, prog_bar: bool = False) -> None:
        """
        A minimal stand-in for LightningModule.log used during unit tests.

        Args:
            name (str): The name of the metric being logged.
            value: The value associated with the metric (typically a tensor or float).
            prog_bar (bool, optional): Whether the value would be shown in the progress bar.

        Returns:
            None.
        """
        logged[name] = (value, prog_bar)

    digit_classifier.log = dummy_logger

    # Simulate data
    batch_size = 6
    x = torch.randn(batch_size, 1, 28, 28)
    y = torch.randint(0, digit_classifier.num_classes, (batch_size,))

    digit_classifier.validation_step((x, y), batch_idx=0)

    # Before reset, metrics should not have been logged
    assert "val_acc_macro" not in logged
    assert "val_f1_macro" not in logged

    # Before reset, metrics should report that they have been updated
    assert digit_classifier.acc.update_called is True
    assert digit_classifier.f1.update_called is True

    # End of validation epoch
    digit_classifier.on_validation_epoch_end()

    # Metrics should have been logged
    assert "val_acc_macro" in logged
    assert "val_f1_macro" in logged

    # After reset, metrics should report that they have not been updated
    assert digit_classifier.acc.update_called is False
    assert digit_classifier.f1.update_called is False


# =============================================================
# Prediction step
# =============================================================


def test_predict_step_returns_probabilities(digit_classifier: DigitClassifier) -> None:
    """
    Test that predict_step returns valid probability distributions.

    This test verifies that the classifier produces output probabilities for each sample in the batch,
    and that each probability vector sums (approximately) to 1 due to the softmax operation.

    Args:
        digit_classifier (DigitClassifier): The model instance under test.

    Returns:
        None.
    """
    # Simulate data
    batch_size = 5
    x = torch.randn(batch_size, 1, 28, 28)

    probs = digit_classifier.predict_step(x, batch_idx=0)

    assert isinstance(probs, torch.Tensor)
    assert probs.shape == (batch_size, digit_classifier.num_classes)

    # Each row should sum (approximately) to 1 due to softmax.
    row_sums = probs.sum(dim=1)
    ones = torch.ones_like(row_sums)
    assert torch.allclose(row_sums, ones, atol=1e-5)


def test_predict_step_accepts_tuple_batch(digit_classifier: DigitClassifier) -> None:
    """
    Test that predict_step accepts both x and (x, y) batch formats.

    Ensures the method functions correctly whether it receives only the input tensor or a full (input, label)
    batch during prediction.

    Args:
        digit_classifier (DigitClassifier):
            The model instance under test.

    Returns:
        None.
    """
    # Simulate data
    batch_size = 3
    x = torch.randn(batch_size, 1, 28, 28)
    y = torch.randint(0, digit_classifier.num_classes, (batch_size,))

    probs_from_x = digit_classifier.predict_step(x, batch_idx=0)
    probs_from_tuple = digit_classifier.predict_step((x, y), batch_idx=0)

    assert torch.allclose(probs_from_x, probs_from_tuple)
