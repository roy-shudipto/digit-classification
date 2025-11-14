import pytest
import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from digit_classification.model import DigitClassifier


# =============================================================
# Fixtures for testing Model
# =============================================================
@pytest.fixture
def index_to_label() -> dict[int, int]:
    """
    Provide a simple mapping from internal class indices to original MNIST labels.

    Returns:
        dict[int, int]: Mapping of internal indices to digit labels.
    """
    return {0: 0, 1: 5, 2: 8}


@pytest.fixture
def digit_classifier(index_to_label: dict[int, int]) -> DigitClassifier:
    """
    Create a DigitClassifier instance without class weights.

    Args:
        index_to_label (dict[int, int]): Mapping of internal class indices to MNIST labels.

    Returns:
        DigitClassifier: Model instance initialized without class weights.
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
    Create a DigitClassifier instance with predefined class weights.

    Args:
        index_to_label (dict[int, int]): Mapping of internal class indices to MNIST labels.

    Returns:
        DigitClassifier: Model instance initialized with class weights.
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


@pytest.fixture
def digit_classifier_no_logging(digit_classifier: DigitClassifier) -> DigitClassifier:
    """
    Disable Lightning logging for cleaner unit test output.

    Args:
        digit_classifier (DigitClassifier): Model instance.

    Returns:
        DigitClassifier: Model instance with logging overridden.
    """
    digit_classifier.log = lambda *args, **kwargs: None

    return digit_classifier


# =============================================================
# Test Model
# =============================================================
def test_configure_optimizers_returns_adam(digit_classifier: DigitClassifier) -> None:
    """
    Verify that configure_optimizers returns a correct optimizer/scheduler setup.

    This ensures that:
      - Adam is used as the optimizer.
      - ReduceLROnPlateau is the scheduler.
      - Lightning stores hyperparameters in self.hparams.

    Args:
        digit_classifier (DigitClassifier): Model instance under test.

    Returns:
        None.
    """
    config = digit_classifier.configure_optimizers()

    # Extract optimizer and scheduler
    optimizer = config["optimizer"]
    scheduler_config = config["lr_scheduler"]
    scheduler = scheduler_config["scheduler"]

    # Check optimizer type
    assert isinstance(optimizer, torch.optim.Adam)

    # Check scheduler type
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    # Check monitored metric name
    assert scheduler_config["monitor"] == "val_f1_macro"

    # Hyperparameters should be tracked by Lightning
    assert digit_classifier.hparams.num_classes == digit_classifier.num_classes
    assert digit_classifier.hparams.lr == digit_classifier.lr


def test_forward_output_shape(digit_classifier: DigitClassifier) -> None:
    """
    Ensure the forward pass returns logits with shape [batch_size, num_classes].

    Args:
        digit_classifier (DigitClassifier): Model instance under test.

    Returns:
        None.
    """
    # Simulate data
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)

    logits = digit_classifier(x)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (batch_size, digit_classifier.num_classes)


def test_forward_with_wrong_input_channels(digit_classifier: DigitClassifier) -> None:
    """
    Ensure the model raises an error when given a 3-channel image instead of 1-channel MNIST input.

    Args:
        digit_classifier (DigitClassifier): Model instance under test.

    Returns:
        None.
    """
    # Incorrect input: 3-channel image instead of expected 1-channel
    x = torch.randn(4, 3, 28, 28)

    with pytest.raises(RuntimeError):
        digit_classifier(x)


def test_training_step_returns_scalar_loss(
    digit_classifier_no_logging: DigitClassifier,
) -> None:
    """
    Verify that training_step returns a scalar tensor loss.

    Args:
        digit_classifier_no_logging (DigitClassifier): Model instance under test.

    Returns:
        None.
    """
    # Simulate data
    batch_size = 8
    x = torch.randn(batch_size, 1, 28, 28)
    y = torch.randint(0, digit_classifier_no_logging.num_classes, (batch_size,))

    loss = digit_classifier_no_logging.training_step((x, y))

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_training_step_uses_class_weights_returns_scalar_loss(
    digit_classifier_no_logging: DigitClassifier,
) -> None:
    """
    Ensure training_step correctly incorporates class weights when they are set.

    Args:
        digit_classifier_no_logging (DigitClassifier): Model instance under test.

    Returns:
        None.
    """
    # Disable Lightning logging to avoid warnings when no Trainer is attached
    digit_classifier_no_logging.class_weights = torch.linspace(
        1.0, 2.0, steps=digit_classifier_no_logging.num_classes
    )

    # Simulate data
    batch_size = 8
    x = torch.randn(batch_size, 1, 28, 28)
    y = torch.randint(0, digit_classifier_no_logging.num_classes, (batch_size,))

    loss = digit_classifier_no_logging.training_step((x, y))

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_backward_pass_computes_gradients(
    digit_classifier_no_logging: DigitClassifier,
) -> None:
    """
    Verify that all trainable parameters receive gradients during backpropagation.

    Args:
        digit_classifier_no_logging (DigitClassifier): Model instance under test.

    Returns:
        None.
    """
    # Simulate data
    batch_size = 8
    x = torch.randn(batch_size, 1, 28, 28)
    y = torch.randint(0, digit_classifier_no_logging.num_classes, (batch_size,))

    digit_classifier_no_logging.zero_grad()
    loss = digit_classifier_no_logging.training_step((x, y))
    loss.backward()

    # Verify gradients exist
    for name, param in digit_classifier_no_logging.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_validation_step_updates_metrics_consistently(
    digit_classifier_no_logging: DigitClassifier,
) -> None:
    """
    Ensure validation_step updates accuracy and F1 metrics consistently with expected values.

    Args:
        digit_classifier_no_logging (DigitClassifier): Model instance under test.

    Returns:
        None.
    """
    # Simulate data
    num_classes = digit_classifier_no_logging.num_classes
    batch_size = 10
    x = torch.randn(batch_size, 1, 28, 28)
    y = torch.randint(0, num_classes, (batch_size,))

    # Compute logits once to derive expected metrics.
    with torch.no_grad():
        logits = digit_classifier_no_logging(x)
    preds = logits.argmax(dim=1)

    # Reference metrics using the same configuration.
    ref_acc = MulticlassAccuracy(num_classes=num_classes, average="macro")
    ref_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
    ref_acc.update(preds, y)
    ref_f1.update(preds, y)

    expected_acc = ref_acc.compute()
    expected_f1 = ref_f1.compute()

    # Run the actual validation step, which should update the model's metrics.
    digit_classifier_no_logging.validation_step((x, y))

    model_acc = digit_classifier_no_logging.acc.compute()
    model_f1 = digit_classifier_no_logging.f1.compute()

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
    logged: dict[str, torch.Tensor] = {}

    def dummy_logger(name: str, value: torch.Tensor | float) -> None:
        """
        A minimal stand-in for LightningModule.log used during unit tests.

        Args:
            name (str): The name of the metric being logged.
            value (torch.Tensor | float): The value associated with the metric (typically a tensor or float).

        Returns:
            None.
        """
        logged[name] = value

    digit_classifier.log = dummy_logger

    # Simulate data
    batch_size = 6
    x = torch.randn(batch_size, 1, 28, 28)
    y = torch.randint(0, digit_classifier.num_classes, (batch_size,))

    digit_classifier.validation_step((x, y))

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

    probs = digit_classifier.predict_step(x)

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
        digit_classifier (DigitClassifier): The model instance under test.

    Returns:
        None.
    """
    # Simulate data
    batch_size = 3
    x = torch.randn(batch_size, 1, 28, 28)
    y = torch.randint(0, digit_classifier.num_classes, (batch_size,))

    probs_from_x = digit_classifier.predict_step(x)
    probs_from_tuple = digit_classifier.predict_step((x, y))

    assert torch.allclose(probs_from_x, probs_from_tuple)
