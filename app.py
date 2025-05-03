import streamlit as st
from transformers import TFRobertaForSequenceClassification, RobertaTokenizer
import tensorflow as tf
import pandas as pd
import os
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer

# Load model and tokenizer

model = TFDistilBertForSequenceClassification.from_pretrained("saved_model_distilbert", local_files_only=True)
tokenizer = DistilBertTokenizer.from_pretrained("saved_model_distilbert", local_files_only=True)

# Set up feedback file
FEEDBACK_FILE = "user_feedback.csv"
if not os.path.exists(FEEDBACK_FILE):
    pd.DataFrame(columns=["text", "score", "feedback"]).to_csv(FEEDBACK_FILE, index=False)

# Streamlit title
st.title("Faux-Finder")
st.write("Enter a news article and get a credibility score.")

# Maintain chat history in session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
input_text = st.chat_input("Write a news article...")

if input_text:
    st.session_state.chat_history.append({"role": "user", "content": input_text})

    # Tokenize and predict
    inputs = tokenizer(input_text, return_tensors="tf", truncation=True, padding=True, max_length=512)
    probs = tf.nn.softmax(model(inputs).logits, axis=1)
    credibility_score = float(probs[0][1].numpy())  # score for "Real"

    # Determine verdict and color
    if credibility_score >= 0.85:
        verdict = "🟢 **Highly credible** – This news is very likely to be real."
        bar_color = "green"
    elif credibility_score >= 0.6:
        verdict = "🟡 **Somewhat credible** – This news might be real, but not certain."
        bar_color = "orange"
    else:
        verdict = "🔴 **Not credible** – This news is likely to be fake."
        bar_color = "red"

    # Save prediction to history
    assistant_message = f"### 🔍 Credibility Score\n**Score**: `{credibility_score:.2f}`\n{verdict}"
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": assistant_message,
        "score": credibility_score,
        "color": bar_color
    })

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            st.markdown(f"""
            <div style="background-color:lightgray; border-radius:8px; height:20px; width:100%;">
                <div style="background-color:{msg['color']}; width:{msg['score']*100}%; height:100%; border-radius:8px;"></div>
            </div>
            """, unsafe_allow_html=True)

# Feedback section
if input_text:
    feedback = st.radio("Was this prediction accurate?", ["✅ Yes", "❌ No"], horizontal=True)
    if st.button("Submit Feedback"):
        df = pd.read_csv(FEEDBACK_FILE)
        df = df.append({
            "text": input_text,
            "score": credibility_score,
            "feedback": feedback
        }, ignore_index=True)
        df.to_csv(FEEDBACK_FILE, index=False)
        st.success("✅ Feedback saved!")
