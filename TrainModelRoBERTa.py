# pip install transformers tensorflow scikit-learn pandas
# Required libraries
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification

# Step 1: Load dataset and keep only relevant columns
df = pd.read_csv(
    "combined_dataset_fake&real.csv",
    encoding="ISO-8859-1",
    on_bad_lines='skip',
    quoting=3
)

# Step 2: Keep only the columns we care about
df = df[["text", "label", "subject", "date"]]

# Step 3: Clean label column
df["label"] = df["label"].astype(str).str.strip()             # Ensure labels are strings
df = df[df["label"].isin(["0", "1"])]                         # Keep only 0 and 1
df["label"] = df["label"].astype(int)                         # Convert back to int

# Step 4: Drop rows with empty or invalid text
df = df.dropna(subset=["text"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

# Step 5: Display info to verify cleaning worked
print("✅ Cleaned dataset")
print("Number of rows:", len(df))
print("Unique labels:", df["label"].unique())
print("Sample rows:")
print(df.sample(3))

texts = df["text"].tolist()
labels = df["label"].tolist()

# Step 2: Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

# Step 3: Tokenize the texts using RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

train_encodings = tokenizer(X_train, truncation=True, padding=True, return_tensors="tf", max_length=512)
test_encodings = tokenizer(X_test, truncation=True, padding=True, return_tensors="tf", max_length=512)

# Step 4: Convert encoded data to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train)).batch(16)
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test)).batch(16)

# Step 5: Load RoBERTa model for binary classification (Fake/Real)
model = TFRobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# Step 6: Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# Step 7: Train the model
history = model.fit(train_dataset, validation_data=test_dataset, epochs=3)

# Step 8: Evaluate the model
y_pred_logits = model.predict(test_dataset).logits
y_pred = np.argmax(y_pred_logits, axis=1)

# Step 9: Print performance metrics
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

print("\n🔀 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

