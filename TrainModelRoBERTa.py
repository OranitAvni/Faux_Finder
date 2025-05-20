import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

# Step 1: Load new full balanced dataset (28,000 rows)
df = pd.read_csv("final_balanced_dataset_with_source.csv")

print("✅ Full dataset loaded!")
print(df["label"].value_counts())
print(df["source"].value_counts())

# Step 2: Prepare data
texts = df["text"].astype(str).tolist()  # convert to string in case of NaN
labels = df["label"].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

# Step 3: Tokenization
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=512, return_tensors="np")
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=512, return_tensors="np")

# Step 4: TF Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": train_encodings["input_ids"], "attention_mask": train_encodings["attention_mask"]},
    tf.convert_to_tensor(y_train)
)).batch(16)

test_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": test_encodings["input_ids"], "attention_mask": test_encodings["attention_mask"]},
    tf.convert_to_tensor(y_test)
)).batch(16)

# Step 5: Load model
model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.distilbert.trainable = False  # Freeze base model

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# Step 6: Early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=1,
    restore_best_weights=True
)

# Step 7: Training
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=5,
    callbacks=[early_stop]
)

# Step 8: Evaluation
y_pred_logits = model.predict(test_dataset).logits
y_pred = np.argmax(y_pred_logits, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

print("\n🔀 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 9: Save model and tokenizer
save_path = "saved_model_distilbert_covid_politics"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"✅ Model and tokenizer saved successfully to folder: {save_path}")


