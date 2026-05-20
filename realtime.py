import cv2
import numpy as np
from tensorflow.keras.models import load_model
 
# =====================
# LOAD MODEL
# =====================
model = load_model("emotion_model.keras", compile=False)
 
# ✅ Must match alphabetical order TF used during training
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
 
emotion_colors = {
    "happy":    (0,   220,   0),
    "sad":      (200,  50,  50),
    "angry":    (0,    0,  255),
    "surprise": (0,   220, 220),
    "fear":     (180,  0,  180),
    "disgust":  (0,   140,   0),
    "neutral":  (160, 160, 160),
}
 
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
 
# =====================
# WEBCAM LOOP
# =====================
cap = cv2.VideoCapture(0)
 
if not cap.isOpened():
    print("ERROR: Cannot open webcam. Check your camera connection.")
    exit()
 
print("Press Q to quit.")
 
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)    # ignore tiny false detections
    )
 
    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]
 
        # ✅ BGR → RGB before feeding to model
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        face_input   = face_resized.astype("float32") / 255.0
        face_input   = np.expand_dims(face_input, axis=0)
 
        preds      = model.predict(face_input, verbose=0)
        idx        = int(np.argmax(preds))
        emotion    = emotion_labels[idx]
        confidence = float(np.max(preds))
 
        color = emotion_colors.get(emotion, (200, 200, 200))
        label = f"{emotion}  {confidence:.0%}"
 
        # Face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
 
        # Label background bar
        cv2.rectangle(frame, (x, y - 35), (x + w, y), color, -1)
        cv2.putText(
            frame, label,
            (x + 6, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2
        )
 
    cv2.imshow("FaceVibe — Realtime Emotion Detection  [Q to quit]", frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()