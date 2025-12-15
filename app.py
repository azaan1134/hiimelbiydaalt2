import os
import json
import base64
import numpy as np
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# ================= ФАЙЛЫН НЭРС =================
MODEL_FILE = "language_model.h5"
META_FILE = "tokenizer_meta.json"
SAVE_DIR = "saved_images"
COUNTER_FILE = "screenshot_counter.txt"  # Дугаарлалтын тоолуур файл

# ================= FLASK ТОХИРГОО =================
app = Flask(__name__)
app.secret_key = "replace-this-with-a-secret"

# Screenshot хадгалах хавтас (одоо ашиглахгүй, зөвхөн нэр үүсгэхэд)
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

# ================= IMAGE FILE NAME (тогтвортой дугаарлалт) =================
def get_next_filename():
    if os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE, 'r') as f:
            current_id = int(f.read().strip())
    else:
        current_id = 0

    next_id = current_id + 1
    with open(COUNTER_FILE, 'w') as f:
        f.write(str(next_id))

    return f"user_{next_id}.png"

# ================= SMTP ГМАИЛ ФУНКЦ =================
def send_to_email(image_bytes, filename, email_to, email_from, app_password):
    try:
        # Имэйл бүтээх
        message = MIMEMultipart()
        message['From'] = email_from
        message['To'] = email_to
        message['Subject'] = f"Таамаглалын зураг: {filename}"

        # Зураг хавсаргах
        attach = MIMEBase('application', 'octet-stream')
        attach.set_payload(image_bytes)
        encoders.encode_base64(attach)
        attach.add_header('Content-Disposition', f'attachment; filename= {filename}')
        message.attach(attach)

        # SMTP сервер холбогдох
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Аюулгүй холболт
        server.login(email_from, app_password)  # Илгээгч имэйл ба app password

        # Имэйл илгээх
        text = message.as_string()
        server.sendmail(email_from, email_to, text)
        server.quit()

        return f"Имэйл илгээгдлээ: {email_to} руу {filename} хавсаргасан."
    except Exception as e:
        return f"Имэйл илгээхэд алдаа: {str(e)}"

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

@app.route("/get_counter", methods=["GET"])
def get_counter():
    if os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE, 'r') as f:
            current_id = int(f.read().strip())
    else:
        current_id = 0
    return jsonify({"count": current_id})

@app.route("/save_image", methods=["POST"])
def save_image():
    data = request.json.get("image")

    if not data or "," not in data:
        return jsonify({"success": False, "error": "Invalid image data"})

    try:
        image_data = data.split(",")[1]
        image_bytes = base64.b64decode(image_data)

        filename = get_next_filename()  # Дугаарлагдсан нэр авах

        # Таны имэйл хаяг (env var-аас авах, Render дээр тохируулах)
        email_to = os.environ.get('EMAIL_TO', 'jargal130613@gmail.com')
        email_from = os.environ.get('EMAIL_FROM', 'azjargala076@gmail.com')
        app_password = os.environ.get('EMAIL_APP_PASSWORD', 'acys zqsv akqv hkxz')

        # Gmail-ээр илгээх
        email_status = send_to_email(image_bytes, filename, email_to, email_from, app_password)
        if "алдаа" in email_status:
            return jsonify({"success": False, "error": email_status})

        print(f"📸 Screenshot sent from {email_from} to {email_to}: {filename}")

        return jsonify({"success": True, "filename": filename, "email_status": email_status})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)