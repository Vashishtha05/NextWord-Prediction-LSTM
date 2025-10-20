# =========================================================
# 🚀 Next Word Prediction using LSTM – Professional UI
# =========================================================

import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Next Word Prediction • LSTM",
    page_icon="🧠",
    layout="centered"
)


# ---------------------------------------------------------
# Load Model + Tokenizer (Cached for Performance)
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    model = load_model("next_word_lstm.h5")
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer


model, tokenizer = load_assets()


# ---------------------------------------------------------
# Custom Styling – Clean Interview-Ready UI
# ---------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(120deg,#0f172a,#111827);
}

.main-title {
    text-align:center;
    font-size:38px;
    font-weight:700;
    color:white;
}

.subtitle {
    text-align:center;
    color:#cbd5f5;
    margin-bottom:25px;
}

.result-box {
    background:#111827;
    border:1px solid #1f2937;
    padding:20px;
    border-radius:12px;
    font-size:22px;
    text-align:center;
    color:#22c55e;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown('<div class="main-title">🧠 LSTM Next Word Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Deep Learning NLP model predicting the next word based on sequence context</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Prediction Logic
# ---------------------------------------------------------
def predict_next_word(model, tokenizer, text, max_sequence_len):

    token_list = tokenizer.texts_to_sequences([text])[0]

    # ensure proper sequence length
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding="pre"
    )

    prediction = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(prediction, axis=1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

    return "Unknown"


# ---------------------------------------------------------
# Input Section (Professional Layout)
# ---------------------------------------------------------
with st.container():

    st.markdown("### ✍️ Enter Text Sequence")

    input_text = st.text_input(
        "",
        placeholder="Example: To be or not to"
    )

    predict_btn = st.button("🔮 Predict Next Word", use_container_width=True)


# ---------------------------------------------------------
# Prediction Output
# ---------------------------------------------------------
if predict_btn:

    if input_text.strip() == "":
        st.warning("Please enter a valid text sequence.")
    else:
        max_sequence_len = model.input_shape[1] + 1
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)

        st.markdown(
            f'<div class="result-box">Predicted Next Word → <b>{next_word}</b></div>',
            unsafe_allow_html=True
        )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption("Built using TensorFlow • LSTM • NLP • Streamlit")
