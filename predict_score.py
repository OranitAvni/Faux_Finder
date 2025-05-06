from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import numpy as np
import tensorflow as tf

# טעינת המודל וה-tokenizer
model = TFDistilBertForSequenceClassification.from_pretrained("saved_model_distilbert")
tokenizer = DistilBertTokenizer.from_pretrained("saved_model_distilbert")

def predict_score(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=512)
    outputs = model(inputs)
    logits = outputs.logits.numpy()[0]
    probs = tf.nn.softmax(logits).numpy()
    real_score = probs[1]  # הסתברות לאמת
    return real_score

# בדיקה ידנית
if __name__ == "__main__":
    while True:
        text = input("הכנס טקסט לבדיקה ('exit' כדי לצאת): ")
        if text.lower() == "exit":
            break
        score = predict_score(text)
        print(f"score: {score:.2f}")
