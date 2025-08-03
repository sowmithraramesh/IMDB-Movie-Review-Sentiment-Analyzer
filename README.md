## 🎬 IMDB Movie Review Sentiment Analyzer
LIVE : [https://moviereview-sentiment-analyzer.streamlit.app/](https://imdb-movie-review-sentiment-analyzer-huwoyflgzffjesxokqtsmc.streamlit.app/)

This project is an End-to-End Sentiment Analysis Pipeline built using PyTorch (Deep Learning) and deployed through Streamlit as an interactive web app.
The model classifies movie reviews from the IMDB dataset into Positive or Negative sentiments.

## 📖 Table of Contents
Overview
Features
Architecture
Dataset
Installation & Usage
Demo
Results
Tech Stack
Future Enhancements
Overview
Movie reviews are an excellent benchmark for Natural Language Processing tasks.
## This project:

Preprocesses text data (cleaning, tokenization, encoding)
Trains an LSTM neural network from scratch using PyTorch
Deploys an interactive web application using Streamlit
The web app lets users input any movie review and instantly get:

Sentiment classification (Positive/Negative)
Confidence score displayed as stars

## Features
🧠 Deep Learning Model: LSTM-based architecture
🗃 Dataset: IMDB 50,000 labeled reviews
🖥 Interactive Streamlit UI with:
Text input for user reviews
Result display with confidence scores
Custom background and styled UI
⭐ Confidence visualization using star ratings
🏆 Model accuracy achieved: ~87%
## Tech Stack
Python 3.9+
PyTorch for deep learning
Streamlit for the interactive web app
Pandas, NumPy, scikit-learn for data handling
## Architecture
Steps:

Data Preprocessing
Vocabulary Encoding
Train/Test Split
Model Training (LSTM)
Save model (imdb_lstm_model.pth)
Deploy on Streamlit
Model Summary:

Embedding Layer
LSTM Layer
Dropout (0.5)
Fully Connected Layer
Sigmoid Output
Dataset
## Dataset used:
IMDB 50K Movie Reviews

25,000 labeled reviews for training
25,000 labeled reviews for testing
Balanced dataset (Positive & Negative reviews)
## Output
<img width="1838" height="996" alt="Screenshot 2025-08-01 230159" src="https://github.com/user-attachments/assets/106c7cce-2409-478f-80a3-2215fa5e166e" />

