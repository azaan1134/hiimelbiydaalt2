import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# ---------- 1. Файл унших ба цэвэрлэх ----------
with open("zuir.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Бүх тэмдэгт устгах, зөвхөн монгол үсэг ба хоосон зай үлдээх
text = text.lower()
text = re.sub(r"[^а-яА-ЯёЁөӨүҮһҺ\s]", " ", text)
text = re.sub(r"\s+", " ", text).strip()

words = text.split()
print(f"Цэвэрлэсэн үгний тоо: {len(words)}, Давтагдашгүй үгс: {len(set(words))}")

# ---------- 2. Vocab үүсгэх ----------
vocab = sorted(set(words))
word_to_index = {w: i for i, w in enumerate(vocab)}
index_to_word = {i: w for i, w in enumerate(vocab)}

# ---------- 3.  болон label үүсгэх ----------
seq_length = 3
sequences = []
labels = []

for i in range(len(words) - seq_length):
    seq = words[i:i+seq_length]
    label = words[i+seq_length]
    sequences.append([word_to_index[w] for w in seq])
    labels.append(word_to_index[label])

X = np.array(sequences)
y = np.array(labels)
y_one_hot = tf.keras.utils.to_categorical(y, num_classes=len(vocab))

# ---------- 4. LSTM загвар ----------
model = Sequential([
    Embedding(input_dim=len(vocab), output_dim=50, input_length=seq_length),
    LSTM(100, activation='tanh'),
    Dense(len(vocab), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---------- 5. Сургалт ----------
model.fit(X, y_one_hot, epochs=50, batch_size=64)

# ---------- 6. Текст таамаглах ----------
start_seq = ["Монгол"]  # Эхлэл үг
generated = start_seq.copy()

for _ in range(10):  # 10 үг таамаглах
    cur = generated[-seq_length:]
    # Хэрвээ хангалтгүй бол <PAD> нэмэх
    if len(cur) < seq_length:
        cur = ["<PAD>"]*(seq_length-len(cur)) + cur
    indices = [word_to_index.get(w, 0) for w in cur]  # <PAD> болон танигдаагүй үгийг 0-р солих
    x = np.array([indices])
    pred = model.predict(x, verbose=0)[0]
    next_index = np.argmax(pred)
    next_word = index_to_word[next_index]
    generated.append(next_word)

print("Generated Text:")
print(" ".join(generated))

# ---------- 7. Загвар хадгалах ----------
model.save("language_model.h5")
print("Saved model: language_model.h5")

meta = {
    "vocab": vocab,
    "seq_length": seq_length
}
with open("tokenizer_meta.json", "w", encoding="utf-8") as f:
    import json
    json.dump(meta, f, ensure_ascii=False)
print("Saved tokenizer metadata: tokenizer_meta.json")
