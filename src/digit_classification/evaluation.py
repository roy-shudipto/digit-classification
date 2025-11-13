import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from typing import Tuple, Dict

from .model import DigitClassifier


def eval_classifier(
    model: DigitClassifier,
    test_loader: DataLoader,
    index_to_label: Dict[int, int],
) -> Tuple[str, torch.Tensor]:
    """
    Evaluate a classifier on a test dataset and return a classification report
    and confusion matrix.

    Args:
        model (DigitClassifier):
            A trained PyTorch DigitClassifier model.

        test_loader (DataLoader):
            A DataLoader yielding batches of (inputs, targets), where targets
            are integer class indices in the range 0 to C-1.

        index_to_label (dict[int, int]):
            Mapping from contiguous model class indices to original dataset labels.
            Example: {0: 0, 1: 5, 2: 8}.

    Returns:
        Tuple[str, torch.Tensor]:
            - A formatted text classification report.
            - A confusion matrix as a torch.Tensor of shape [C, C].
    """
    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            logits = model(inputs)
            predicted = logits.argmax(dim=1)

            true_labels.extend(targets.cpu().tolist())
            pred_labels.extend(predicted.cpu().tolist())

    # Class indices used by the model
    label_indices = sorted(index_to_label.keys())

    # Convert model-class indices to original digit names for display
    target_names = [str(index_to_label[i]) for i in label_indices]

    report = classification_report(
        true_labels,
        pred_labels,
        labels=label_indices,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    cm_np = confusion_matrix(true_labels, pred_labels, labels=label_indices)
    cm = torch.tensor(cm_np, dtype=torch.int64)

    return report, cm
