import streamlit as st

st.title("🤖 My Custom AI Chatbot")

# Inisialisasi history chat agar tidak hilang saat refresh
if "messages" not in st.session_state:
    st.session_state.messages = []

# Menampilkan chat lama
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input user
if prompt := st.chat_input("Tanya sesuatu..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Masukkan logika model AI kamu di sini
    # response = my_ai_model.predict(prompt)
    response = "Ini adalah jawaban dari model buatanmu sendiri!"

    with st.chat_message("assistant"):
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
      
