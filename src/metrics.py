from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np


def compute_classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """Compute accuracy and macro-F1 score.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary with accuracy and macro_f1 keys.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def compute_per_class_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str] | None = None,
) -> dict:
    """Compute detailed per-class metrics including confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Optional class names for display.

    Returns:
        Dictionary with per-class precision, recall, f1, and confusion matrix.
    """
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)
    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
