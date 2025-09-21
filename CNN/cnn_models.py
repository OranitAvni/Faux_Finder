# CNN/cnn_models.py
"""
Generic 1D text CNN builder for binary text classification.

Usage (in your training script):
    from CNN.cnn_models import build_text_cnn

    model = build_text_cnn(
        vocab_size=len(vocab),
        seq_len=150,
        embed_dim=100,
        conv_blocks=[(64, 2, 3), (128, 2, 3)],
        dropout=0.3,
        classifier_units=1,
        classifier_activation="sigmoid",
        embedding_weights=embedding_matrix,   # or None
        embedding_trainable=True,
        global_pool="max",
        name="cnn_text_cls",
    )
"""

from typing import Iterable, Optional, Tuple
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, MaxPooling1D,
    GlobalMaxPooling1D, GlobalAveragePooling1D,
    Dropout, Dense
)


def build_text_cnn(
        *,
        vocab_size: int,
        seq_len: int,
        embed_dim: int,
        conv_blocks: Iterable[Tuple[int, int, int]],
        dropout: float = 0.0,
        classifier_units: int = 1,
        classifier_activation: Optional[str] = "sigmoid",
        embedding_weights: Optional["np.ndarray"] = None,  # noqa: F821 (type hint for numpy)
        embedding_trainable: bool = True,
        global_pool: str = "max",  # "max" or "avg"
        name: str = "text_cnn",
) -> Model:
    """
    Build a generic text CNN with configurable conv blocks.

    Args:
        vocab_size: size of the vocabulary (Embedding input_dim).
        seq_len: fixed sequence length.
        embed_dim: embedding dimension.
        conv_blocks: iterable of (filters, kernel_size, pool_size).
        dropout: dropout rate applied after blocks and before head.
        classifier_units: number of output units (1 for binary).
        classifier_activation: 'sigmoid' (binary) or None for logits.
        embedding_weights: optional pre-trained embedding matrix (vocab_size x embed_dim).
        embedding_trainable: whether embedding is trainable.
        global_pool: 'max' or 'avg'.
        name: model name.
    """
    tokens = Input(shape=(seq_len,), dtype="int32", name="tokens")

    # Embedding
    if embedding_weights is not None:
        x = Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            weights=[embedding_weights],
            trainable=embedding_trainable,
            name="embed",
        )(tokens)
    else:
        x = Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            name="embed",
        )(tokens)

    # Conv blocks
    for idx, (filters, ksize, psize) in enumerate(conv_blocks, start=1):
        x = Conv1D(filters, ksize, activation="relu", name=f"conv{idx}")(x)
        if psize and psize > 1:
            x = MaxPooling1D(psize, name=f"pool{idx}")(x)
        if dropout and dropout > 0:
            x = Dropout(dropout, name=f"drop{idx}")(x)

    # Global pooling
    if global_pool == "avg":
        x = GlobalAveragePooling1D(name="gpool")(x)
    else:
        x = GlobalMaxPooling1D(name="gpool")(x)

    if dropout and dropout > 0:
        x = Dropout(dropout, name="drop_head")(x)

    outputs = Dense(classifier_units, activation=classifier_activation, name="classifier")(x)
    return Model(tokens, outputs, name=name)
