import numpy as np
import tensorflow as tf
import cv2
import os

# Load your existing model
model = tf.keras.models.load_model("emotion_model.keras", compile=False)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# =====================
# FER-2013 class distribution (approximate)
# These numbers tell us how biased the dataset is
# =====================
class_counts = {
    'angry':    4953,
    'disgust':   547,   # ← tiny! model ignores this
    'fear':     5121,
    'happy':    8989,   # ← huge! model over-predicts this
    'neutral':  6198,
    'sad':      6077,
    'surprise': 4002,
}

total = sum(class_counts.values())

# Prior probability of each class in the dataset
priors = np.array([class_counts[e] for e in emotion_labels], dtype=np.float32)
priors = priors / priors.sum()

print("Dataset priors (how skewed the data is):")
for label, p in zip(emotion_labels, priors):
    print(f"  {label}: {p:.3f}")

# =====================
# CORRECTED PREDICT FUNCTION
# Divides model output by dataset prior so rare emotions
# get a fair chance — no retraining needed!
# =====================
def predict_corrected(image_bgr):
    """
    image_bgr: any OpenCV image (BGR format)
    Returns: (emotion_string, confidence_float, all_scores_dict)
    """
    # Preprocess
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (48, 48))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Raw model output
    raw = model.predict(img, verbose=0)[0]

    # ✅ Correction: divide by prior, re-normalize
    corrected = raw / (priors + 1e-8)
    corrected = corrected / corrected.sum()

    idx        = int(np.argmax(corrected))
    emotion    = emotion_labels[idx]
    confidence = float(corrected[idx])
    all_scores = {emotion_labels[i]: round(float(corrected[i]), 4) for i in range(7)}

    return emotion, confidence, all_scores


# =====================
# QUICK TEST on a solid-color image just to verify all classes work
# =====================
print("\nSanity check — feeding random noise images:")
emotions_seen = set()
for _ in range(50):
    noise = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    emo, conf, _ = predict_corrected(noise)
    emotions_seen.add(emo)

print(f"Emotions predicted on noise: {emotions_seen}")
print("(Should see multiple emotions, not just angry)")
print("\n✅ fix_predictions.py ready. Import predict_corrected() in app.py")
