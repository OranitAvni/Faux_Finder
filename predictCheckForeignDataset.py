import pandas as pd
import tensorflow as tf
import numpy as np
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score

# Load trained model and tokenizer
model_path = "saved_model_distilbert"
model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Load dataset to evaluate
df = pd.read_csv("evaluation_dataset.csv")
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
