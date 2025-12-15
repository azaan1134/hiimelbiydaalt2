import os
import json
import base64
import numpy as np
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
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

# ================= GMAIL API ФУНКЦ =================
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def send_to_email(image_bytes, filename, email_to):
    creds = None
    # Render дээр env var ашиглах
    if 'GOOGLE_CREDENTIALS' in os.environ:
        creds_json = json.loads(os.environ['GOOGLE_CREDENTIALS'])  # JSON string
        flow = InstalledAppFlow.from_client_config(creds_json, SCOPES)
    else:
        # Локал дээр credentials.json ашиглах
        if os.path.exists('credentials.json'):
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        else:
            return "credentials.json алга байна! Google Cloud-аас татаж авна уу."

    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            creds = flow.run_local_server(port=0)  # Эхний удаа browser нээгдэж Gmail зөвшөөрөх
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('gmail', 'v1', credentials=creds)
    message = MIMEMultipart()
    message['to'] = email_to
    message['subject'] = f"Таамаглалын зураг: {filename}"
    attach = MIMEBase('application', 'octet-stream')
    attach.set_payload(image_bytes)
    encoders.encode_base64(attach)
    attach.add_header('Content-Disposition', f'attachment; filename= {filename}')
    message.attach(attach)
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    body = {'raw': raw}
    service.users().messages().send(userId='me', body=body).execute()
    return f"Имэйл илгээгдлээ: {email_to} руу {filename} хавсаргасан."

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

        # Таны имэйл хаяг
        email_to = "jargal130613@gmail.com"

        # Gmail API-ээр илгээх (SMTP биш)
        email_status = send_to_email(image_bytes, filename, email_to)
        if "алга байна" in email_status:
            return jsonify({"success": False, "error": email_status})

        print(f"📸 Screenshot sent to {email_to}: {filename}")

        return jsonify({"success": True, "filename": filename, "email_status": email_status})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)