# 🎬 IMDB Movie Review Sentiment Analyzer

## Project Overview

This project analyzes sentiment in movie reviews using the **IMDB 50K dataset**—classifying each review as either **positive** or **negative**. It includes:

- **Data preprocessing**: cleaning text (HTML stripping, stop-word removal, normalization)
- **Tokenization & padding**: converting reviews into padded sequences for input
- **Word embeddings**: using GloVe vectors to capture semantic context
- **Modeling**: exploring multiple architectures such as:
  - Simple feed-forward neural network
  - **Convolutional Neural Network (CNN)**
  - **Long Short-Term Memory (LSTM)** network
- **Evaluation**: comparing performance (accuracy, precision, recall, F1) of each model
- **User interface**: optional demo app (e.g., via Flask or Streamlit) to input reviews and see sentiment predictions

The goal is to demonstrate how deep learning-based NLP models can effectively classify textual sentiment and to compare model architectures.

---

## Steps to Run Locally

1. **Clone the repository**  
   ```bash
   git clone https://github.com/sowmithraramesh/IMDB-Movie-Review-Sentiment-Analyzer.git
   cd IMDB-Movie-Review-Sentiment-Analyzer
2.Set up a Python environment (recommended)

bash
Copy code
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
3.Install dependencies

bash
Copy code
pip install -r requirem
ents.txt

## Results Highlights
LSTM models consistently outperform simpler architectures for sentiment classification on this dataset, often achieving accuracy in the mid to high 80% range 
irjmets.com
github.com
<img width="1838" height="996" alt="Screenshot 2025-08-01 230159" src="https://github.com/user-attachments/assets/e7af71f1-6618-4066-951a-e7e1fcec3811" />

CNNs and feed-forward networks also provide reasonable performance but slightly behind LSTMs

This project offers hands‑on insights into text preprocessing, tokenization, embedding, and neural model tuning in NLP contexts


