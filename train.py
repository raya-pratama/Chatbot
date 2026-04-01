import json
import numpy as np
import nltk
import pickle
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from nltk.stem import LancasterStemmer

# Download data pendukung NLTK (Tambahkan punkt_tab)
nltk.download('punkt')
nltk.download('punkt_tab') # BARIS BARU INI UNTUK MEMPERBAIKI LOOKUPERROR
stemmer = LancasterStemmer()

# --- SISANYA SAMA SEPERTI KODE SEBELUMNYA ---

# 1. DATASET
intents_data = {
  "intents": [
    {"tag": "salam", "patterns": ["Halo", "Hai", "P", "Selamat pagi", "Selamat siang"], "responses": ["Halo! Ada yang bisa dibantu?", "Hai juga!"]},
    {"tag": "tanya_kabar", "patterns": ["Apa kabar?", "Gimana kabarmu?", "Sehat?"], "responses": ["Aku baik, aku cuma bot tapi aku senang membantu!"]},
    {"tag": "perpisahan", "patterns": ["Dah", "Bye", "Sampai jumpa", "Terima kasih", "Makasih"], "responses": ["Sampai jumpa!", "Sama-sama, senang bisa membantu!"]}
  ]
}

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# 2. PREPROCESSING
for intent in intents_data['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [stemmer.stem(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# 3. MENGATUR DATA TRAINING
random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# 4. MEMBANGUN MODEL (NEURAL NETWORK)
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# 5. SIMPAN HASILNYA
model.save('chatbot_model.h5')
with open("data.pkl", "wb") as f:
    pickle.dump({'words': words, 'classes': classes}, f)

print("\n--- BERHASIL! ---")
print("Silakan download 'chatbot_model.h5' dan 'data.pkl' dari folder di kiri.")
