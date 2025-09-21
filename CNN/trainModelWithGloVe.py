# CNN/trainModelWithGloVe.py
# Clean, generic CNN training script using shared utils and a generic model builder.

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import TextVectorization
from keras.optimizers import Adam

from common.data_utils import load_text_cls_splits
from common.viz_utils import plot_loss
from common.eval_utils import to_labels, confusion_and_report
from CNN.cnn_models import build_text_cnn  # <- external generic builder

# --- Step 0: (optional) reproducibility ---
SEED = 42
tf.keras.utils.set_random_seed(SEED)

# --- Step 1: Load data ---
X_train, X_test, y_train, y_test = load_text_cls_splits(
    csv_path="../data/balanced_10k_dataset_politics.csv",
    text_col="text",
    label_col="label",
    test_size=0.2,
    seed=SEED
)

# --- Step 2: Vectorization ---
MAX_VOCAB = 20_000
SEQ_LEN = 150

vectorizer = TextVectorization(
    max_tokens=MAX_VOCAB,
    output_sequence_length=SEQ_LEN
)
vectorizer.adapt(tf.data.Dataset.from_tensor_slices(X_train).batch(256))

def to_ds(texts, labels, batch=32):
    X = vectorizer(tf.constant(list(texts)))           # -> int32 [N, SEQ_LEN]
    y = tf.constant(labels, dtype=tf.float32)          # -> float32 [N]
    return (tf.data.Dataset
            .from_tensor_slices((X, y))
            .batch(batch)
            .prefetch(tf.data.AUTOTUNE))

train_ds = to_ds(X_train, y_train)
test_ds  = to_ds(X_test,  y_test)

# --- Step 3: Build model (generic) ---
# Use the vocabulary the vectorizer actually learned:
vocab = vectorizer.get_vocabulary()
V = len(vocab)        # true vocab size (<= MAX_VOCAB)
EMB_DIM = 100

# If you have a GloVe embedding matrix aligned to 'vocab', pass it here.
embedding_matrix = None  # replace with your matrix (shape [V, EMB_DIM]) if available.

model = build_text_cnn(
    vocab_size=V,
    seq_len=SEQ_LEN,
    embed_dim=EMB_DIM,
    conv_blocks=[(64, 2, 3), (128, 2, 3)],   # your specific architecture
    dropout=0.3,
    classifier_units=1,
    classifier_activation="sigmoid",
    embedding_weights=embedding_matrix,
    embedding_trainable=True,
    global_pool="max",
    name="cnn_text_cls"
)

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# --- Step 4: Train ---
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=30,
    callbacks=[early_stop]
)

# --- Step 5: Plot loss ---
plot_loss(history, title="CNN Loss")

# --- Step 6: Evaluate ---
# IMPORTANT: Predict over tokenized inputs (not raw strings)
X_test_vec = vectorizer(tf.constant(X_test))           # shape [N, SEQ_LEN], int32
y_pred_probs = model.predict(X_test_vec, batch_size=32).reshape(-1)
y_pred = to_labels(y_pred_probs, threshold=0.5)

confusion_and_report(y_test, y_pred, target_names=("Fake", "Real"))

# (optional) also print Keras evaluate for loss/accuracy agreement:
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"\nKeras evaluate → loss={test_loss:.4f}  acc={test_acc:.4f}")

# --- Step 7: Save model ---
model.save("CNN_Models/kaggle_dataset_politics_after_changes.keras", save_format="keras")
print("✅ Saved to CNN_Models/kaggle_dataset_politics.keras")
