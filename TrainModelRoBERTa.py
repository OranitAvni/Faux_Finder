from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
import tensorflow as tf
import numpy as np

# טען את המודל וה-tokenizer ששמרת
model = TFDistilBertForSequenceClassification.from_pretrained("saved_model_distilbert")
tokenizer = DistilBertTokenizer.from_pretrained("saved_model_distilbert")

# ודאי שהשכבות הלא־מאומנות עדיין קפואות (אם צריך)
model.distilbert.trainable = False

# טקסט חדש לבדיקה
new_text = "As U.S. budget fight looms, Republicans flip their fiscal script"

# בצעי טוקניזציה
inputs = tokenizer(new_text, return_tensors="tf", truncation=True, padding=True, max_length=512)

# ניבוי
outputs = model(inputs)
logits = outputs.logits
probs = tf.nn.softmax(logits, axis=-1).numpy()[0]  # המרה להסתברויות

# קחי את ההסתברות למחלקה 1 (Real)
score = probs[1]  # מחלקה 1 = אמיתי (בהנחה שזה הסדר שהיה באימון)

print(f"🧠 Prediction score: {score:.2f}")
if score > 0.5:
    print("✅ This looks like REAL news")
else:
    print("⚠️ This looks like FAKE news")
