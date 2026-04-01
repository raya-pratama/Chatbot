import streamlit as st
import numpy as np
import pickle
import nltk
from tensorflow.keras.models import load_model
from nltk.stem import LancasterStemmer

# Setup NLTK
nltk.download('punkt')
nltk.download('punkt_tab')
stemmer = LancasterStemmer()

st.set_page_config(page_title="AI Chatbot Portofolio", page_icon="🤖")

# --- CUSTOM CSS UNTUK TAMPILAN KANAN KIRI ---
st.markdown("""
    <style>
    /* Container untuk chat user (Kanan) */
    .user-container {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 10px;
    }
    .user-bubble {
        background-color: #007bff;
        color: white;
        padding: 10px 15px;
        border-radius: 20px 20px 0px 20px;
        max-width: 70%;
        text-align: right;
    }
    /* Container untuk chat bot (Kiri) */
    .bot-container {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 10px;
    }
    .bot-bubble {
        background-color: #f1f1f1;
        color: black;
        padding: 10px 15px;
        border-radius: 20px 20px 20px 0px;
        max-width: 70%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL & ASSETS ---
@st.cache_resource
def load_assets():
    model = load_model('chatbot_model.h5')
    data = pickle.load(open("data.pkl", "rb"))
    return model, data['words'], data['classes']

try:
    model, words, classes = load_assets()
except:
    st.error("Pastikan file 'chatbot_model.h5' dan 'data.pkl' sudah ada di GitHub!")
    st.stop()

# Jawaban Bot (Sesuai Tag)
responses_dict = {
    "salam": "Halo! Ada yang bisa saya bantu hari ini?",
    "tanya_kabar": "Saya merasa luar biasa! Bagaimana dengan Anda?",
    "perpisahan": "Sampai jumpa! Datang lagi ya kalau butuh bantuan.",
    "terima_kasih": "Sama-sama! Senang bisa membantu."
}

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [stemmer.stem(word.lower()) for word in sentence_words]

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: bag[i] = 1
    return np.array(bag)

# --- LOGIKA CHAT ---
st.title("🤖 My Custom AI Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Menampilkan riwayat chat dengan format CSS
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-container"><div class="user-bubble">{message["content"]}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-container"><div class="bot-bubble">{message["content"]}</div></div>', unsafe_allow_html=True)

# Input User
if prompt := st.chat_input("Ketik pesan..."):
    # Tampilkan user ke kanan
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="user-container"><div class="user-bubble">{prompt}</div></div>', unsafe_allow_html=True)

    # Prediksi AI
    input_data = bow(prompt, words)
    prediction = model.predict(np.array([input_data]), verbose=0)[0]
    results_index = np.argmax(prediction)
    tag = classes[results_index]
    
    if prediction[results_index] > 0.5:
        reply = responses_dict.get(tag, "Saya mengerti, tapi belum ada jawaban pasti.")
    else:
        reply = "Maaf, saya tidak mengerti maksud Anda."

    # Tampilkan bot ke kiri
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.markdown(f'<div class="bot-container"><div class="bot-bubble">{reply}</div></div>', unsafe_allow_html=True)
    
