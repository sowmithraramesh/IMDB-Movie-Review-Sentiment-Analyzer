import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
import numpy as np
import base64

# ---------- FUNCTION TO ENCODE LOCAL IMAGE ----------
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load local background image and encode it
background_image = "background.jpg"  # Ensure this file is in the same folder
encoded_image = get_base64_image(background_image)

# ---------- CUSTOM CSS ----------
custom_css = f"""
<style>
.stApp {{
    background-image: url("data:image/jpg;base64,{encoded_image}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

header {{visibility: hidden;}}

.main-card {{
    background: rgba(0, 0, 0, 0.65);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    max-width: 700px;
    margin: auto;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}}

.title-text {{
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: white;
    text-shadow: 0px 0px 8px #ff66ff, 0px 0px 15px #00ffff;
}}

textarea {{
    background-color: rgba(0,0,0,0.5) !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid #888 !important;
}}

.stButton>button {{
    background: #28a745;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    border: none;
    transition: 0.3s;
}}
.stButton>button:hover {{
    background: #218838;
    transform: scale(1.05);
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---------- LOAD MODEL AND VOCAB ----------
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

MAX_LEN = 200
device = torch.device("cpu")

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, output_dim=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size+1, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        out, (h, c) = self.lstm(x)
        h = self.dropout(h[-1])
        out = self.fc(h)
        return self.sigmoid(out)

model = SentimentLSTM(len(vocab))
model.load_state_dict(torch.load("imdb_lstm_model.pth", map_location=device))
model.eval()

# ---------- TEXT PREPROCESS ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text

def encode(text):
    return [vocab.get(w, 0) for w in text.split()]

def pad(seq, max_len=MAX_LEN):
    return seq[:max_len] + [0]*(max_len-len(seq))

# ---------- UI ----------
# st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.markdown("<h1 class='title-text'>🎬 Movie Review Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Enter a review below to analyze sentiment.</p>", unsafe_allow_html=True)

user_input = st.text_area("Enter your review:", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        encoded = encode(cleaned)
        padded = pad(encoded)
        tensor = torch.tensor([padded], dtype=torch.long).to(device)

        with st.spinner("Analyzing..."):
            with torch.no_grad():
                pred = model(tensor)
                score = pred.item()
                sentiment = "Positive 😊" if score > 0.5 else "Negative 😞"

        # Results
        st.subheader("Prediction Result")
        stars = int(round(score * 5))
        star_display = "⭐" * stars + "☆" * (5 - stars)
        color = "lightgreen" if score > 0.5 else "salmon"

        st.markdown(f"<h3 style='color:{color}; text-align:center;'>{sentiment}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:18px; text-align:center;'>Confidence: {star_display} ({score:.2f})</p>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
