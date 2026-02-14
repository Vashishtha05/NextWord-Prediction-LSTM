# ✨ LSTM Next Word Prediction — Neural Language Model

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-LSTM-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/NLP-Sequence%20Modeling-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Streamlit-Interactive%20App-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/Python-AI%20Project-yellow?style=for-the-badge">
</p>

<p align="center">
  🚀 A Deep Learning NLP project that predicts the next word in a sentence using an LSTM-based neural language model.
</p>

---

## 📌 Overview

This project demonstrates **Neural Language Modeling** using Long Short-Term Memory (LSTM) networks.
The model learns sequential text patterns and predicts the most probable next word given an input phrase.

The application includes:

* 🧠 Deep Learning LSTM architecture
* 📚 NLP tokenization & sequence preprocessing
* ⚡ Real-time predictions using Streamlit
* 🔎 Practical example of sequence modeling for intelligent text generation

---

## ✨ Features

* 🔮 Next-word prediction using trained LSTM
* 🧩 Tokenizer-based text preprocessing
* 📊 Early stopping training strategy
* ⚡ Fast interactive web interface
* 🧠 Demonstrates practical NLP + Deep Learning skills

---

## ⚙️ Tech Stack

| Technology         | Usage                   |
| ------------------ | ----------------------- |
| Python             | Core Programming        |
| TensorFlow / Keras | Deep Learning Model     |
| LSTM               | Sequence Prediction     |
| NLP Tokenizer      | Text Processing         |
| Streamlit          | Interactive UI          |
| Pickle             | Tokenizer Serialization |

---

## 📂 Project Structure

```
LSTM RNN/
│
├── app.py                          # Streamlit application
├── experiments.ipynb               # Training experiments
├── hamlet.txt                      # Training dataset
├── next_word_lstm.h5               # Trained LSTM model
├── tokenizer.pickle                # Saved tokenizer
├── requirements.txt                # Dependencies
```

---

## 🧠 Model Architecture

* Embedding & tokenization
* LSTM layers for sequence learning
* Dense output layer for word prediction
* Early stopping for optimized training

The model learns contextual patterns and generates intelligent text continuations.

---

## 🚀 Getting Started

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/LSTM-Next-Word-Predictor.git
cd LSTM-Next-Word-Predictor
```

### 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

## 🖥️ How It Works

1. User enters a text sequence.
2. Text is tokenized and padded.
3. LSTM predicts probability distribution over vocabulary.
4. Highest probability word is returned as next prediction.

---

## 🧪 Example

**Input:**

```
To be or not to
```

**Prediction:**

```
be
```

---

## 📌 Future Improvements

* Transformer-based language models
* Top-K word prediction
* Temperature-based text generation
* Full sentence auto-completion
* Deployment on cloud platforms

---

## 👨‍💻 Author

**Vashishtha Verma**

* 🤖 Machine Learning & Generative AI
* 🧠 Agentic AI Systems
* 💻 Software Engineering & DSA

