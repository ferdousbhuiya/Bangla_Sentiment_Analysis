import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Set page configuration
st.set_page_config(page_title="Bangla Sentiment Analysis", page_icon="🇧🇩")

# Title
st.title("🇧🇩 Bangla Sentiment Analysis")
st.markdown("Analyze the sentiment (Positive, Neutral, Negative) of Bangla text using a pretrained BERT model.")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
    model = AutoModelForSequenceClassification.from_pretrained("csebuetnlp/banglabert")
    return tokenizer, model

tokenizer, model = load_model()

# Text input
text = st.text_area("✍️ Enter Bangla text:", "")

# Analyze button
if st.button("🔍 Analyze Sentiment") and text.strip():
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # Define sentiment labels
    label_map = {0: "Negative 😞", 1: "Neutral 😐", 2: "Positive 😊"}
    st.success(f"**Predicted Sentiment:** {label_map.get(predicted_class, 'Unknown')}")


