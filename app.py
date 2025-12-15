import os
import json
import base64
import numpy as np
from flask import Flask, request, render_template, jsonify
import tensorflow as tf

# ================= ФАЙЛЫН НЭРС =================
MODEL_FILE = "language_model.h5"
META_FILE = "tokenizer_meta.json"
SAVE_DIR = "saved_images"

# ================= FLASK ТОХИРГОО =================
app = Flask(__name__)
app.secret_key = "replace-this-with-a-secret"

# Screenshot хадгалах хавтас
os.makedirs(SAVE_DIR, exist_ok=True)

# ================= ЗАГВАР АЧААЛАХ =================
model = None
vocab = None
seq_length = 2
word_to_index = None
index_to_word = None

if os.path.exists(MODEL_FILE) and os.path.exists(META_FILE):
    model = tf.keras.models.load_model(MODEL_FILE)

    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    vocab = meta["vocab"]
    seq_length = meta.get("seq_length", 3)

    word_to_index = {w: i for i, w in enumerate(vocab)}
    index_to_word = {i: w for i, w in enumerate(vocab)}

    print("✅ Model loaded.")
else:
    print("❌ Model or tokenizer meta file not found.")

# ================= ТЕКСТ ҮҮСГЭХ =================
def generate_text(start_seq, num_words=10):
    if model is None or vocab is None:
        return "Model байхгүй байна."

    generated = start_seq.split()

    for _ in range(num_words):
        cur = generated[-seq_length:]

        if len(cur) < seq_length:
            cur = ["<PAD>"] * (seq_length - len(cur)) + cur

        indices = [
            word_to_index.get(w, word_to_index.get("<UNK>", 0))
            for w in cur
        ]

        x = np.array([indices])
        pred = model.predict(x, verbose=0)[0]

        next_idx = int(np.argmax(pred))
        generated.append(index_to_word[next_idx])

    return " ".join(generated)

# ================= IMAGE FILE NAME =================
def get_next_filename():
    files = [
        f for f in os.listdir(SAVE_DIR)
        if f.startswith("user_") and f.endswith(".png")
    ]

    nums = []
    for f in files:
        try:
            nums.append(int(f.replace("user_", "").replace(".png", "")))
        except ValueError:
            pass

    next_id = max(nums) + 1 if nums else 1
    return f"user_{next_id}.png"

# ================= ROUTES =================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    start_seq = request.form.get("start_seq", "").strip()
    if not start_seq:
        return "Эхлэл текст оруулна уу."

    try:
        num_words = int(request.form.get("num_words", 10))
    except ValueError:
        num_words = 10

    return generate_text(start_seq, num_words)

@app.route("/save_image", methods=["POST"])
def save_image():
    data = request.json.get("image")

    if not data or "," not in data:
        return jsonify({"success": False, "error": "Invalid image data"})

    try:
        image_data = data.split(",")[1]
        image_bytes = base64.b64decode(image_data)

        filename = get_next_filename()
        path = os.path.join(SAVE_DIR, filename)

        with open(path, "wb") as f:
            f.write(image_bytes)

        print(f"📸 Screenshot saved: {filename}")

        return jsonify({"success": True, "filename": filename})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)