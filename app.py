import streamlit as st
import numpy as np
import pickle
import nltk
import os
from tensorflow.keras.models import load_model
from nltk.stem import LancasterStemmer

# Download resource NLTK
nltk.download('punkt')
nltk.download('punkt_tab')
stemmer = LancasterStemmer()

# Konfigurasi Halaman
st.set_page_config(page_title="AI Chatbot Portofolio", page_icon="🤖")

# --- FUNGSI UTAMA AI ---
@st.cache_resource
def load_assets():
    # Load model dan data pendukung dari hasil training Colab
    model = load_model('chatbot_model.h5')
    data = pickle.load(open("data.pkl", "rb"))
    return model, data['words'], data['classes']

try:
    model, words, classes = load_assets()
except Exception as e:
    st.error("File 'chatbot_model.h5' atau 'data.pkl' tidak ditemukan. Pastikan sudah upload hasil training dari Colab!")
    st.stop()

# Dataset jawaban (Sesuaikan tag-nya dengan yang ada di Colab)
responses_dict = {
    "salam": "Halo! Senang bertemu denganmu. Ada yang bisa dibantu?",
    "tanya_kabar": "Aku baik-baik saja! Sebagai AI, aku selalu siap membantumu.",
    "perpisahan": "Sampai jumpa lagi! Semoga harimu menyenangkan.",
    "terima_kasih": "Sama-sama! Senang bisa berguna untukmu."
}

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# --- TAMPILAN UI ---
st.title("🤖 My Custom Neural Network Chatbot")
st.markdown("---")

# Inisialisasi Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Menampilkan riwayat chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input User
if prompt := st.chat_input("Ketik sesuatu (misal: Halo)"):
    # Simpan dan tampilkan pesan user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prediksi AI
    input_data = bow(prompt, words)
    prediction = model.predict(np.array([input_data]), verbose=0)[0]
    
    # Ambil index dengan probabilitas tertinggi
    results_index = np.argmax(prediction)
    tag = classes[results_index]
    
    # Ambang batas keyakinan (Confidence Threshold)
    if prediction[results_index] > 0.5:
        reply = responses_dict.get(tag, "Aku mengerti maksudmu, tapi aku belum punya jawaban spesifik untuk itu.")
    else:
        reply = "Maaf, aku belum mempelajari kata itu. Bisa coba bahasa lain?"

    # Tampilkan jawaban Bot
    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
        
