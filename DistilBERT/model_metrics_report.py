from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("../data/combined_balanced_dataset_with_source.csv")
texts = df["text"].astype(str).tolist()
labels = df["label"].tolist()

# Split data (must match original training)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

# Load tokenizer & model
model_path = "saved_model_distilbert_covid_politics"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = TFDistilBertForSequenceClassification.from_pretrained(model_path)

# Tokenize test data
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=512, return_tensors="tf")

# Create test dataset
test_dataset = tf.data.Dataset.from_tensor_slices((
    {
        "input_ids": test_encodings["input_ids"],
        "attention_mask": test_encodings["attention_mask"]
    }
)).batch(16)

# Predict
y_pred_logits = model.predict(test_dataset).logits
y_pred = np.argmax(y_pred_logits, axis=1)

# Confusion matrix
print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("📊 Confusion Matrix Details:")
print(f"True Positives (TP):  {tp}   --> Correctly predicted REAL news")
print(f"False Positives (FP): {fp}   --> Predicted REAL, but was FAKE")
print(f"False Negatives (FN): {fn}   --> Predicted FAKE, but was REAL")
print(f"True Negatives (TN):  {tn}   --> Correctly predicted FAKE news")
