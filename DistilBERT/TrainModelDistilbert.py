import tensorflow as tf
import numpy as np
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

from common.data_utils import load_text_cls_splits
from common.viz_utils import plot_loss
from common.eval_utils import to_labels, confusion_and_report

# --- Step 1: Load data ---
X_train, X_test, y_train, y_test = load_text_cls_splits(
    csv_path="../data/balanced_from_self_and_zenodo-covid19.csv",
    text_col="text",
    label_col="label",
    outcome_col="outcome",
    outcome_map={"fake": 0, "real": 1},
    test_size=0.2,
    seed=42
)

# --- Step 2: Tokenization ---
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=512, return_tensors="tf")
test_enc = tokenizer(X_test, truncation=True, padding=True, max_length=512, return_tensors="tf")

train_ds = tf.data.Dataset.from_tensor_slices((dict(train_enc), tf.constant(y_train))).batch(16).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((dict(test_enc), tf.constant(y_test))).batch(16).prefetch(tf.data.AUTOTUNE)

# --- Step 3: Build model ---
model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.distilbert.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# --- Step 4: Train ---
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)
history = model.fit(train_ds, validation_data=test_ds, epochs=5, callbacks=[early_stop])

# --- Step 5: Plot loss ---
plot_loss(history, title="DistilBERT Loss")

# --- Step 6: Evaluate ---
y_pred_logits = model.predict(test_ds).logits
y_pred = to_labels(y_pred_logits, from_logits=True)
confusion_and_report(y_test, y_pred, target_names=("Fake", "Real"))

# --- Step 7: Save model ---
save_path = "saved_model_distilbert_covid_after_changes"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
