import streamlit as st
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the pretrained emotion detection model and tokenizer
model_name = "distilbert-finetuned-emotion"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define a function to predict emotion
def predict_emotion(text):
    input_encoded = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**input_encoded)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).item()
    return pred

# Streamlit app title
st.title('Emotion Detection with Transformers')

# Input text area for user input
input_text = st.text_area('Enter text for emotion detection:', 'I feel happy')

# Button to trigger emotion detection
if st.button('Detect Emotion'):
    # Predict the emotion
    predicted_emotion_index = predict_emotion(input_text)
    st.subheader('Emotion Detected:')
    st.write(classes[predicted_emotion_index])

if __name__ == "__main__":
    st.run()
