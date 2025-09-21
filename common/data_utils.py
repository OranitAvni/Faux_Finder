import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional, Dict

def load_text_cls_splits(
        csv_path: str,
        text_col: str = "text",
        label_col: str = "label",
        outcome_col: Optional[str] = None,
        outcome_map: Optional[Dict[str, int]] = None,
        test_size: float = 0.2,
        seed: int = 42,
) -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    Load CSV and split into train/test sets.
    If 'outcome_col' is provided, map its values using 'outcome_map' into 'label_col'.
    Returns X_train, X_test, y_train, y_test.
    """
    df = pd.read_csv(csv_path)

    if outcome_col is not None and outcome_map is not None:
        df[label_col] = df[outcome_col].map(outcome_map)

    df = df.dropna(subset=[text_col])
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    return X_train, X_test, y_train, y_test
