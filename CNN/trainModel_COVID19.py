# CNN/trainModel_COVID19.py
# Clean, generic CNN training script for 'balanced_dataset.csv' (headlines/outcome).

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from keras.optimizers import Adam

from common.data_utils import load_text_cls_splits
from common.viz_utils import plot_loss
from common.eval_utils import to_labels, confusion_and_report
from CNN.cnn_models import build_text_cnn  # generic CNN builder (you created in CNN/cnn_models.py)

# -----------------------
# Reproducibility (optional)
# -----------------------
SEED = 42
tf.keras.utils.set_random_seed(SEED)

# -----------------------
# 1) Load & split data
# -----------------------
# If your 'outcome' column is already 0/1, we can use it directly.
# If it's "fake"/"real", uncomment outcome_map below and add outcome_col="outcome".
X_train, X_test, y_train, y_test = load_text_cls_splits(
    csv_path="balanced_dataset.csv",
    text_col="headlines",
    label_col="outcome",
    outcome_col=None,                 # set to "outcome" if you need mapping
    outcome_map=None,                 # e.g., {"fake": 0, "real": 1}
    test_size=0.2,
    seed=SEED
)

# -----------------------
# 2) Text vectorization
# -----------------------
MAX_VOCAB = 20_000
SEQ_LEN   = 150

vectorizer = TextVectorization(
    max_tokens=MAX_VOCAB,
    output_sequence_length=SEQ_LEN
)
vectorizer.adapt(tf.data.Dataset.from_tensor_slices(X_train).batch(256))

def to_ds(texts, labels, batch=32):
    X = vectorizer(tf.constant(list(texts)))      # int32 [N, SEQ_LEN]
    y = tf.constant(labels, dtype=tf.float32)     # float32 [N]
    return (tf.data.Dataset
            .from_tensor_slices((X, y))
            .batch(batch)
            .prefetch(tf.data.AUTOTUNE))

train_ds = to_ds(X_train, y_train, batch=32)
test_ds  = to_ds(X_test,  y_test,  batch=32)

# -----------------------
# 3) Build model (generic)
# -----------------------
vocab = vectorizer.get_vocabulary()
V = len(vocab)
EMB_DIM = 20  # you used 20 in the original script

model = build_text_cnn(
    vocab_size=V,
    seq_len=SEQ_LEN,
    embed_dim=EMB_DIM,
    conv_blocks=[                 # mirrors your original architecture
        (32, 3, 3),
        (64, 3, 3),
        (128, 3, 0),             # last block had no pooling in your code
    ],
    dropout=0.3,                  # you used Dropout after the first block
    classifier_units=1,
    classifier_activation="sigmoid",
    embedding_weights=None,       # set a matrix here if you add GloVe later
    embedding_trainable=True,
    global_pool="max",
    name="cnn_headlines_cls"
)

model.compile(
    optimizer=Adam(learning_rate=1e-3),  # same as 'adam' default; change if needed
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -----------------------
# 4) Train
# -----------------------
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

# -----------------------
# 5) Plot loss
# -----------------------
plot_loss(history, title="CNN Loss (headlines/outcome)")

# -----------------------
# 6) Evaluate
# -----------------------
# Predict on the tokenized test inputs (NOT raw strings)
X_test_vec = vectorizer(tf.constant(X_test))
y_pred_probs = model.predict(X_test_vec, batch_size=32).reshape(-1)
y_pred = to_labels(y_pred_probs, threshold=0.5)

confusion_and_report(y_test, y_pred, target_names=("Fake", "Real"))

# Optional: also show Keras evaluate
loss, acc = model.evaluate(test_ds, verbose=0)
print(f"\nKeras evaluate → loss={loss:.4f}  acc={acc:.4f}")

# -----------------------
# 7) Save model
# -----------------------
model.save("CNN_Models/covid_headlines.keras", save_format="keras")
print("✅ Saved to CNN_Models/covid_headlines.keras")
