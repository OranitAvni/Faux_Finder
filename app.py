
import streamlit as st
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
import tensorflow as tf
import pandas as pd
import os

# Load model and tokenizer
model = TFDistilBertForSequenceClassification.from_pretrained("saved_model_distilbert", local_files_only=True)
tokenizer = DistilBertTokenizer.from_pretrained("saved_model_distilbert", local_files_only=True)

# Feedback file setup
FEEDBACK_FILE = "user_feedback.csv"
if not os.path.exists(FEEDBACK_FILE):
    pd.DataFrame(columns=["text", "score", "feedback"]).to_csv(FEEDBACK_FILE, index=False)

# Page UI
# st.title("Faux-Finder")
st.image("logo.png", width=400)
st.subheader("Evaluate the credibility of a news article")

# Custom CSS for styling
st.markdown("""
    <style>
    .stTextArea textarea {
        font-size: 18px !important;
        min-height: 80px !important;
        max-height: 300px !important;
        overflow-y: auto;
    }
    </style>
""", unsafe_allow_html=True)

# Text input - dynamic height look
input_text = st.text_area("Enter a news headline or short article:")

# Analyze button - always visible
analyze_clicked = st.button("Analyze")

# Run prediction if button clicked
if analyze_clicked and input_text.strip():
    inputs = tokenizer(input_text, return_tensors="tf", truncation=True, padding=True, max_length=512)
    probs = tf.nn.softmax(model(inputs).logits, axis=1)
    credibility_score = float(probs[0][1].numpy())

    # Determine verdict
    if credibility_score >= 0.85:
        verdict = "Highly credible – This news is very likely to be real."
        bar_color = "green"
    elif credibility_score >= 0.6:
        verdict = "Somewhat credible – This news might be real, but not certain."
        bar_color = "orange"
    else:
        verdict = "Not credible – This news is likely to be fake."
        bar_color = "red"

    # Display results
    st.markdown(f"### Credibility Score: `{credibility_score:.2f}`")
    st.markdown(verdict)
    st.markdown(f"""
        <div style="background-color:#e0e0e0; border-radius:5px; height:20px; width:100%; margin-top:10px;">
            <div style="background-color:{bar_color}; width:{credibility_score*100}%; height:100%; border-radius:5px;"></div>
        </div>
    """, unsafe_allow_html=True)

    # Feedback section
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
