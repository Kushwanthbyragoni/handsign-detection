import base64
import json
import os
import traceback

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file, abort
from flask_cors import CORS
from tensorflow.keras.models import load_model

# ----------------- PATHS -----------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))         # .../src
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))      # project root
MODEL_DIR = os.path.join(ROOT_DIR, "model")
DATA_DIR = os.path.join(ROOT_DIR, "data")
TEMPLATES_DIR = os.path.join(ROOT_DIR, "templates")
STATIC_DIR = os.path.join(ROOT_DIR, "static")

IMG_SIZE = 160  

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATES_DIR)
CORS(app)

# ----------------- Load model & labels -----------------
MODEL_CANDIDATES = [
    os.path.join(MODEL_DIR, "asl_model.keras"),
    os.path.join(MODEL_DIR, "asl_model.h5"),
    os.path.join(MODEL_DIR, "best_model.h5"),
]

MODEL_PATH = None
for p in MODEL_CANDIDATES:
    if os.path.exists(p):
        MODEL_PATH = p
        break

if MODEL_PATH is None:
    raise FileNotFoundError(
        "No trained model found in 'model'. Expected one of:\n"
        + "\n".join(MODEL_CANDIDATES)
    )

LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_mapping.json")
if not os.path.exists(LABEL_MAP_PATH):
    raise FileNotFoundError(f"Label mapping not found at {LABEL_MAP_PATH}")

print(f"[app.py] Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)

with open(LABEL_MAP_PATH, "r") as f:
    label_mapping = json.load(f)


rev_mapping = {int(v): k for k, v in label_mapping.items()}

# ----------------- MediaPipe Hands -----------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ----------------- Helper functions -----------------
def decode_image_from_data_url(data_url: str):
    
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url
    img_bytes = base64.b64decode(encoded)
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def encode_image_to_base64(img: np.ndarray) -> str:
    """Encode BGR image to base64 JPEG (no data URL header)."""
    _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buf).decode("utf-8")

def get_sample_image_for_label(label: str):
    
    candidate_dirs = [
        os.path.join(DATA_DIR, "Gesture Image Data", label),
        os.path.join(DATA_DIR, "Gesture Image Pre-Processed Data", label),
    ]
    for class_dir in candidate_dirs:
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                return os.path.join(class_dir, fname)
    return None

# ----------------- Routes -----------------
@app.route("/")
def index():
    try:
        reference_labels = sorted(label_mapping.keys(), key=lambda x: str(x).upper())
        return render_template("index.html", reference_labels=reference_labels)
    except Exception:
        tb = traceback.format_exc()
        return f"<h3>Error rendering index.html:</h3><pre>{tb}</pre>", 500

@app.route("/reference/<label>")
def reference_image(label):
    path = get_sample_image_for_label(label)
    if path is None:
        abort(404)
    ext = os.path.splitext(path)[1].lower()
    mimetype = "image/png" if ext == ".png" else "image/jpeg"
    return send_file(path, mimetype=mimetype)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON: { "image": "data:image/jpeg;base64,...." }

    Returns JSON:
    {
      "success": True,
      "prediction": "<label or 'No hand detected'>",
      "confidence": 0.92,
      "processed_image": "<base64 of full frame with GREEN lines>"
    }
    """
    try:
        data = request.get_json(force=True)
        if not data or "image" not in data:
            return jsonify(success=False, error="No image provided"), 400

        frame = decode_image_from_data_url(data["image"])
        if frame is None:
            return jsonify(success=False, error="Invalid image"), 400

        # Copy for drawing green lines
        draw_frame = frame.copy()

        # Hand detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        prediction_text = "No hand detected"
        confidence = 0.0

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw landmarks and connections in green
            mp_drawing.draw_landmarks(
                draw_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            )

            
            h, w, _ = frame.shape
            xs, ys = [], []
            for lm in hand_landmarks.landmark:
                xs.append(int(lm.x * w))
                ys.append(int(lm.y * h))
            pad = 30
            x_min = max(min(xs) - pad, 0)
            x_max = min(max(xs) + pad, w - 1)
            y_min = max(min(ys) - pad, 0)
            y_max = min(max(ys) + pad, h - 1)
            if x_max > x_min and y_max > y_min:
                cv2.rectangle(draw_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            
            resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
            X = np.expand_dims(img_rgb, axis=0)  # (1,160,160,3)

            preds = model.predict(X)
            confidence = float(np.max(preds))
            idx = int(np.argmax(preds))
            prediction_text = rev_mapping.get(idx, "Unknown")

            # draw label
            label_str = f"{prediction_text} ({confidence*100:.1f}%)"
            cv2.putText(
                draw_frame,
                label_str,
                (max(0, x_min), max(30, y_min - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            prediction_text = "No hand detected"
            confidence = 0.0

        processed_b64 = encode_image_to_base64(draw_frame)

        return jsonify(
            success=True,
            prediction=prediction_text,
            confidence=confidence,
            processed_image=processed_b64,
        )
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify(success=False, error=str(e), traceback=tb), 500


if __name__ == "__main__":
    # Run from src/:  cd HandSignDetection-main/src && python app.py
    app.run(host="0.0.0.0", port=5000, debug=True)
