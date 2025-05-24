import pandas as pd
import tensorflow as tf
import numpy as np
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load trained model and tokenizer
model_path = "../DistilBERT/saved_model_distilbert_Twitter_Articles"
model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Load dataset to evaluate
df = pd.read_csv("final_eval_dataset.csv")
texts = df["text"].tolist()
labels = df["label"].tolist()

# Tokenize all texts
encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="tf")

# Run model on all texts
outputs = model(encodings)
logits = outputs.logits
probs = tf.nn.softmax(logits, axis=1).numpy()
predictions = np.argmax(probs, axis=1)

# Evaluate performance
print("✅ Accuracy:", accuracy_score(labels, predictions))
print("\n📊 Classification Report:")
print(classification_report(labels, predictions, target_names=["Fake", "Real"]))

# Calculate Confusion Matrix
cm = confusion_matrix(labels, predictions)
tn, fp, fn, tp = cm.ravel()

print("\n🔍 Confusion Matrix Values:")
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Negatives (TN): {tn}")
