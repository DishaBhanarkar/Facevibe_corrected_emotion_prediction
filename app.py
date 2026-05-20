from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "emotion_model.keras")
model = load_model(MODEL_PATH, compile=False)

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

@app.route("/")
def home():
    return send_from_directory(os.path.dirname(__file__), "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file       = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img        = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return jsonify({"emotion": "No face detected", "confidence": 0})

    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    (x, y, w, h) = faces[0]
    face = img[y:y+h, x:x+w]

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (48, 48))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)

    # Plain prediction — no correction tricks
    preds      = model.predict(face, verbose=0)[0]
    idx        = int(np.argmax(preds))
    label      = emotion_labels[idx]
    confidence = float(preds[idx])
    all_scores = {emotion_labels[i]: round(float(preds[i]), 4) for i in range(7)}

    return jsonify({
        "emotion":    label,
        "confidence": round(confidence, 4),
        "all_scores": all_scores
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)