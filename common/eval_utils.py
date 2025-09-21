import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from typing import Sequence, Tuple

def to_labels(
        y_pred_raw: np.ndarray,
        threshold: float = 0.5,
        from_logits: bool = False
) -> np.ndarray:
    """
    Convert model outputs into binary labels {0,1}.
    - For CNN: predictions are probabilities → apply threshold.
    - For DistilBERT: predictions are logits/probs of shape (N,2) → take argmax.
    """
    y = np.array(y_pred_raw)
    if y.ndim == 2:
        return np.argmax(y, axis=1)
    return (y >= threshold).astype(int)

def confusion_and_report(
        y_true: Sequence[int],
        y_pred_labels: Sequence[int],
        target_names: Tuple[str, str] = ("Fake", "Real")
) -> dict:
    """
    Print classification report and confusion matrix.
    Returns a dictionary with TN, FP, FN, TP.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred_labels)

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=list(target_names), zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    out = {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "confusion_matrix": cm.tolist()
    }
    print("\n🔀 Confusion Matrix:")
    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    return out
