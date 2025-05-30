# import pandas as pd
# import tensorflow as tf
# import numpy as np
# from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
#
# # Load trained model and tokenizer
# model_path = "../DistilBERT/saved_model_distilbert_Twitter_Articles"
# model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
# tokenizer = DistilBertTokenizer.from_pretrained(model_path)
#
# # Load dataset to evaluate
# df = pd.read_csv("final_eval_dataset.csv")
# texts = df["text"].tolist()
# labels = df["label"].tolist()
#
# # Tokenize all texts
# encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="tf")
#
# # Run model on all texts
# outputs = model(encodings)
# logits = outputs.logits
# probs = tf.nn.softmax(logits, axis=1).numpy()
# predictions = np.argmax(probs, axis=1)
#
# # Evaluate performance
# print("✅ Accuracy:", accuracy_score(labels, predictions))
# print("\n📊 Classification Report:")
# print(classification_report(labels, predictions, target_names=["Fake", "Real"]))
#
# # Calculate Confusion Matrix
# cm = confusion_matrix(labels, predictions)
# tn, fp, fn, tp = cm.ravel()
#
# print("\n🔍 Confusion Matrix Values:")
# print(f"True Positives (TP): {tp}")
# print(f"False Positives (FP): {fp}")
# print(f"False Negatives (FN): {fn}")
# print(f"True Negatives (TN): {tn}")

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
df = pd.read_csv("../data/final_eval_dataset.csv")
texts = df["text"].tolist()
labels = df["label"].tolist()

# Tokenize all texts
encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="tf")

# Run model on all texts
outputs = model(encodings)
logits = outputs.logits
probs = tf.nn.softmax(logits, axis=1).numpy()
predictions = np.argmax(probs, axis=1)

# Flip labels to treat "Fake" (originally 0) as the positive class (1)
labels_adjusted = 1 - np.array(labels)
predictions_adjusted = 1 - predictions

# Accuracy remains unchanged
print("✅ Accuracy:", accuracy_score(labels_adjusted, predictions_adjusted))

# Classification report with Fake as positive class
print("\n📊 Classification Report (Fake = Positive class):")
print(classification_report(labels_adjusted, predictions_adjusted, target_names=["Real", "Fake"], zero_division=0))

# Confusion matrix with flipped labels
tn, fp, fn, tp = confusion_matrix(labels_adjusted, predictions_adjusted).ravel()

print("\n🔍 Confusion Matrix (Fake = Positive class):")
print(f"✅ True Positives (TP – Fake correctly identified):     {tp}")
print(f"❌ False Positives (FP – Real misclassified as Fake):  {fp}")
print(f"❌ False Negatives (FN – Fake misclassified as Real):  {fn}")
print(f"✅ True Negatives (TN – Real correctly identified):     {tn}")
