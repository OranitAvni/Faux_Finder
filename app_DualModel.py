import streamlit as st
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from newspaper import Article

# === Load models ===
model_bert = TFDistilBertForSequenceClassification.from_pretrained(
    "DistilBERT/saved_model_distilbert_politics", local_files_only=True)

tokenizer_bert = DistilBertTokenizer.from_pretrained(
    "DistilBERT/saved_model_distilbert_politics", local_files_only=True)

model_cnn = tf.keras.models.load_model("CNN/politics_model.keras")

# === Feedback file setup ===
FEEDBACK_FILE = "user_feedback.csv"
if not os.path.exists(FEEDBACK_FILE):
    pd.DataFrame(columns=["text", "score", "feedback"]).to_csv(FEEDBACK_FILE, index=False)

# === Custom CSS ===
st.markdown("""<style>html, body, [class*="css"] {
    font-family: 'Arial', sans-serif; font-size: 18px; }
    .stTextArea textarea, .stTextInput input { font-size: 18px !important; }
    .stButton button { font-size: 20px !important; width: 100%; padding: 12px; }
    .stRadio label { font-size: 18px !important; }
    </style>""", unsafe_allow_html=True)

def get_score_color(score):
    if score >= 0.85: return "green"
    elif score >= 0.6: return "orange"
    else: return "red"

# === UI ===
st.image("logo.png", width=400)
st.subheader("Evaluate the credibility of a news article")

input_option = st.radio("Choose input type:", ["Paste text", "Provide URL"], horizontal=True)
input_text = ""

if input_option == "Paste text":
    input_text = st.text_area("Enter a news headline or short article:", height=150)
else:
    url = st.text_input("Enter the URL of the news article:")
    if url.strip():
        try:
            article = Article(url)
            article.download()
            article.parse()
            input_text = article.title + "\n\n" + article.text
            st.success("Article extracted successfully.")
            st.text_area("Extracted article:", input_text, height=200)
        except Exception as e:
            st.error("Failed to extract article. Please check the URL.")

# === Analyze button ===
if st.button("Analyze") and input_text.strip():
    # DistilBERT score
    inputs_bert = tokenizer_bert(input_text, return_tensors="tf", truncation=True, padding=True, max_length=512)
    probs_bert = tf.nn.softmax(model_bert(inputs_bert).logits, axis=1)
    score_bert = float(probs_bert[0][1].numpy())  # Probability of being real

    # CNN score (TextVectorization inside model)
    probs_cnn = model_cnn.predict(tf.constant([input_text]), verbose=0)
    score_cnn = float(probs_cnn[0][0])  # Output shape: (1, 1)

    # Average score
    credibility_score = (score_bert + score_cnn) / 2

    # Display result
    st.markdown("### Credibility Score:")
    position_percent = round((1 - credibility_score) * 100, 2)
    score_color = get_score_color(credibility_score)

    heatbar_html = f"""
    <div style="position: relative; height: 90px; margin-top: 30px; margin-bottom: 20px;">
        <div style="height: 20px; background: linear-gradient(to right, green, yellow, orange, red);
                    border-radius: 10px; box-shadow: inset 0 0 5px #aaa;"></div>
        <div style="position: absolute; top: 25px; left: {position_percent}%;
                    transform: translateX(-50%); text-align: center;">
            <div style="font-size: 24px; color: {score_color}; line-height: 20px;">▲</div>
            <div style="font-size: 16px; font-weight: bold; color: {score_color};">{credibility_score:.2f}</div>
        </div>
    </div>
    """
    st.markdown(heatbar_html, unsafe_allow_html=True)

    # Verdict
    if credibility_score >= 0.85:
        verdict = "Highly credible – This news is very likely to be real."
    elif credibility_score >= 0.6:
        verdict = "Somewhat credible – This news might be real, but not certain."
    else:
        verdict = "Not credible – This news is likely to be fake."
    st.markdown(f"**{verdict}**")

    # Feedback
    feedback = st.radio("Was this prediction accurate?", ["Yes", "No"], horizontal=True)
    if st.button("Submit Feedback"):
        df = pd.read_csv(FEEDBACK_FILE)
        df = pd.concat([df, pd.DataFrame([{
            "text": input_text,
            "score": credibility_score,
            "feedback": feedback
        }])], ignore_index=True)
        df.to_csv(FEEDBACK_FILE, index=False)
        st.success("Feedback saved.")
