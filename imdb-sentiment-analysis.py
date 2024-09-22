import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the model and tokenizer
model = load_model('imdb_sentiment_model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Streamlit UI
st.title("IMDb Review Sentiment Analysis")

review = st.text_area("Enter your review:")

if st.button("Analyze"):
    # Preprocess the input review
    sequences = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(sequences, maxlen=200)  # Adjust maxlen as per your training
    prediction = model.predict(padded)
    
    sentiment = 'Positive' if prediction[0] > 0.5 else 'Negative'
    st.write(f"Sentiment: {sentiment}")
