import streamlit as st
import json
import numpy as np
import pickle
import nltk
from tensorflow.keras.models import load_model
from nltk.stem import LancasterStemmer

# Download data NLTK yang diperlukan server
nltk.download('punkt')
nltk.download('punkt_tab')
stemmer = LancasterStemmer()

# 1. LOAD MODEL DAN DATA
# Gunakan st.cache_resource agar tidak loading terus setiap user ngetik
@st.cache_resource
def load_chat_assets():
    model = load_model('chatbot_model.h5')
    data = pickle.load(open("data.pkl", "rb"))
    words = data['words']
    classes = data['classes']
    # Kita buat dataset respons sederhana di sini (sesuaikan dengan intents di Colab)
    responses = {
        "salam": ["Halo! Ada yang bisa dibantu?", "Hai juga!"],
        "tanya_kabar": ["Aku baik, aku cuma bot tapi aku senang membantu!"],
        "perpisahan": ["Sampai jumpa!", "Sama-sama!"]
    }
    return model, words, classes, responses

model, words, classes, responses = load_chat_assets()

# 2. FUNGSI MEMBERSIHKAN INPUT USER
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

# 3. TAMPILAN STREAMLIT
st.title("🤖 My Neural Network Chatbot")
st.caption("Chatbot ini dibuat dari nol tanpa API!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ketik pesan di sini..."):
    # Tampilkan pesan user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # PREDIKSI DARI MODEL AI
    input_data = bow(prompt, words)
    res = model.predict(np.array([input_data]))[0]
    
    # Ambil index dengan probabilitas tertinggi
    results_index = np.argmax(res)
    tag = classes[results_index]
    
    # Ambil jawaban acak berdasarkan tag (jika akurasi > 0.5)
    if res[results_index] > 0.5:
        reply = responses.get(tag, ["Maaf, aku kurang paham."])[0]
    else:
        reply = "Maaf, aku belum mempelajari itu."

    # Tampilkan jawaban bot
    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
                
