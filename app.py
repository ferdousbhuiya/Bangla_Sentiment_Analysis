import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

st.set_page_config(page_title="Bangla Sentiment Analysis", page_icon="🇧🇩")
st.title("🇧🇩 Bangla Sentiment Analysis")
st.markdown("Analyze the sentiment (positive/neutral/negative) of Bangla text using a pretrained BERT model.")

@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-sentiment")
        model = AutoModelForSequenceClassification.from_pretrained("sagorsarker/bangla-bert-sentiment")
        return tokenizer, model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None

tokenizer, model = load_model()

text = st.text_area("✍️ Enter Bangla text:", "")

if st.button("🔍 Analyze") and text.strip():
    if tokenizer and model:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        labels = {0: "Negative 😞", 1: "Neutral 😐", 2: "Positive 😊"}
        st.success(f"**Prediction:** {labels[pred]}")
    else:
        st.warning("Model could not be loaded. Please try again later.")

