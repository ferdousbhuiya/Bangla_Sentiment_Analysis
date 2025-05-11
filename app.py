import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Title
st.title("Bangla Sentiment Analysis 🇧🇩")
st.markdown("This app uses a pretrained BERT model to predict sentiment of Bangla text.")

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("sagorsarker/bangla-bert-sentiment")
    return tokenizer, model

tokenizer, model = load_model()

# Input
user_input = st.text_area("✍️ Enter Bangla text here:", "")

# Predict sentiment
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some Bangla text.")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

        label_map = {0: "Negative 😞", 1: "Neutral 😐", 2: "Positive 😊"}
        st.success(f"**Predicted Sentiment:** {label_map[predicted_class]}")
