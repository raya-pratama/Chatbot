import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load model yang sudah dibuat di train.py
model = load_model('chatbot_model.h5')

st.set_page_config(page_title="My AI Chatbot", page_icon="🤖")
st.title("Custom Neural Network Chatbot")

# Inisialisasi chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input dari user
if prompt := st.chat_input("Ketik pesan di sini..."):
    # 1. Tampilkan pesan user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Logika Prediksi Model AI kamu
    # Di sini nanti kamu panggil fungsi untuk memproses input dan memprediksi tag
    response = "Ini adalah respon hasil prediksi model buatanmu!"

    # 3. Tampilkan pesan bot
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    
