import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.api.layers import TextVectorization
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Text cleaning
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# Step 1: Load the trained model
model = load_model("CNN_Models/kaggle_dataset_politics.keras")

# Step 2: Load the new dataset
df = pd.read_csv("Datasets/combined_balanced_sport.csv")
df["text"] = df["text"].astype(str).apply(clean_text)

# Step 3: Prepare vectorization
vectorization = TextVectorization(max_tokens=20000, output_sequence_length=150)
vectorization.adapt(df["text"])

# Tensors
text_label_ds = tf.data.Dataset.from_tensor_slices((df["text"], df["label"])).batch(32)
text_ds = tf.data.Dataset.from_tensor_slices(df["text"]).batch(32)

# Compute loss
loss, _ = model.evaluate(text_label_ds, verbose=0)
print(f"🧮 Loss: {loss:.4f}")

# Step 5: Prediction
probs = model.predict(text_ds)
df["score"] = probs.flatten()
df["predicted_label"] = (df["score"] >= 0.5).astype(int)

# Step 6: Evaluation
if "label" in df.columns:
    y_true = 1 - df["label"].astype(int).values
    y_pred = 1 - df["predicted_label"].values

    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    # Mapping for Excel-compatible output
    FN_excel = TP  # Fakes correctly identified
    TP_excel = FP  # Real news classified as Fake
    FP_excel = FN  # Fakes classified as Real
    TN_excel = TN  # Real news correctly identified

    # Metrics
    precision = TP_excel / (TP_excel + FP_excel) if (TP_excel + FP_excel) > 0 else 0
    recall = TP_excel / (TP_excel + FN_excel) if (TP_excel + FN_excel) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP_excel + TN_excel) / (TP_excel + TN_excel + FP_excel + FN_excel)

    # --- Excel-compatible output ---
    print("\n📊 Results compatible with your Excel table:")
    print(f"✅ True Positives (I): {TP_excel}")
    print(f"❌ False Positives (H): {FP_excel}")
    print(f"❌ False Negatives (J): {FN_excel}")
    print(f"✅ True Negatives (K): {TN_excel}")

    print("\n📈 Metrics (columns L–O):")
    print(f"📊 Accuracy (L):  {accuracy:.4f}")
    print(f"🎯 Precision (M): {precision:.4f}")
    print(f"🔁 Recall (N):    {recall:.4f}")
    print(f"💡 F1 Score (O):  {f1:.4f}")
else:
    print("⚠️ 'label' column not found in dataset — cannot compute evaluation metrics.")
