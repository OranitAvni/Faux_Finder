# import pandas as pd
# import tensorflow as tf
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
# from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
#
# # Step 1: Load the cleaned dataset
# df = pd.read_csv("combined_dataset_fake&real_cleaned.csv")
# samples_per_class = 5000  # Half of 5000
#
# # Sample balanced dataset
# fake_df = df[df["label"] == 0].drop_duplicates(subset="text").sample(n=samples_per_class, random_state=42)
# real_df = df[df["label"] == 1].drop_duplicates(subset="text").sample(n=samples_per_class, random_state=42)
#
#
# # Combine and shuffle
# balanced_df = pd.concat([fake_df, real_df]).sample(frac=1, random_state=42).reset_index(drop=True)
#
# print("✅ Balanced dataset created!")
# print(balanced_df["label"].value_counts())
#
# # Step 2: Prepare text and labels
# texts = balanced_df["text"].tolist()
# labels = balanced_df["label"].tolist()
#
# # Step 3: Split into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(
#     texts, labels, test_size=0.2, stratify=labels, random_state=42
# )
#
# # # Step 4: Tokenize the texts using distilroberta tokenizer
# # tokenizer = RobertaTokenizer.from_pretrained("distilroberta-base")
#
# tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
#
# train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=512, return_tensors="np")
# test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=512, return_tensors="np")
#
# # Step 5: Build TensorFlow datasets
# train_dataset = tf.data.Dataset.from_tensor_slices((
#     {
#         "input_ids": train_encodings["input_ids"],
#         "attention_mask": train_encodings["attention_mask"]
#     },
#     tf.convert_to_tensor(y_train)
# )).batch(16)
#
# test_dataset = tf.data.Dataset.from_tensor_slices((
#     {
#         "input_ids": test_encodings["input_ids"],
#         "attention_mask": test_encodings["attention_mask"]
#     },
#     tf.convert_to_tensor(y_test)
# )).batch(16)
#
# # Step 6: Load a lightweight RoBERTa model
# # model = TFRobertaForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2
#
# # model = TFDistilBertForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2)
# model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
#
#
#
# # # Freeze all base layers (only classification head will be trained)
# # model.roberta.trainable = False
# model.distilbert.trainable = False
#
#
# # Step 7: Compile the model
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=["accuracy"]
# )
#
# # Step 8: Train the model with early stopping
# early_stop = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     patience=1,
#     restore_best_weights=True
# )
#
# history = model.fit(train_dataset, validation_data=test_dataset, epochs=5, callbacks=[early_stop])
#
# # Step 9: Evaluate the model
# y_pred_logits = model.predict(test_dataset).logits
# y_pred = np.argmax(y_pred_logits, axis=1)
#
# print("\n📊 Classification Report:")
# print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))
#
# print("\n🔀 Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
#
# # Step 10: Save the trained model and tokenizer
# model.save_pretrained("saved_model_distilbert")
# tokenizer.save_pretrained("saved_model_distilbert")
#
# model.save("saved_model_distilbert/tf_model.h5")
#
#
# print("✅ Model and tokenizer saved to folder 'saved_model_distilbert'")
#

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

# Step 1: Load and balance dataset
df = pd.read_csv("combined_dataset_fake&real_cleaned.csv")
samples_per_class = 5000

fake_df = df[df["label"] == 0].drop_duplicates(subset="text").sample(n=samples_per_class, random_state=42)
real_df = df[df["label"] == 1].drop_duplicates(subset="text").sample(n=samples_per_class, random_state=42)
balanced_df = pd.concat([fake_df, real_df]).sample(frac=1, random_state=42).reset_index(drop=True)

print("✅ Balanced dataset created!")
print(balanced_df["label"].value_counts())

# Step 2: Prepare data
texts = balanced_df["text"].tolist()
labels = balanced_df["label"].tolist()

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, stratify=labels, random_state=42)

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

# Step 6: Early stopping only — remove ModelCheckpoint (incompatible with this model class)
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

# Step 9: Save model and tokenizer in HuggingFace format (for app.py)
save_path = "saved_model_distilbert"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"✅ Model and tokenizer saved successfully to folder: {save_path}")
