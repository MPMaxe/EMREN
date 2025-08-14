from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import numpy as np
import onnxruntime as ort
import io
import os

# --- paths (relative to THIS file) ---
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "resnet18.onnx")
LABELS_PATH = os.path.join(HERE, "labels.txt")

# --- load model & labels once at startup ---
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
with open(LABELS_PATH, "r") as f:
    LABELS = [x.strip() for x in f if x.strip()]

app = Flask(__name__, static_folder=HERE, static_url_path="")

def preprocess(pil_img):
    # resize to 224x224, normalize like torchvision ResNet
    pil_img = pil_img.resize((224, 224))
    arr = np.array(pil_img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std                 # HWC
    arr = np.transpose(arr, (2, 0, 1))       # CHW
    arr = np.expand_dims(arr, axis=0)        # NCHW
    return arr

@app.route("/", methods=["GET"])
def index():
    # serve the page from the same origin -> no CORS issues
    return send_from_directory(HERE, "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400

    file = request.files["file"]
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"bad image: {e}"}), 400

    x = preprocess(img)
    inputs = {session.get_inputs()[0].name: x}
    logits = session.run(None, inputs)[0]          # shape [1, num_classes]
    # softmax to probabilities
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    pred_idx = int(np.argmax(probs, axis=1)[0])
    confidence = float(probs[0, pred_idx])

    return jsonify({
        "emotion": LABELS[pred_idx] if pred_idx < len(LABELS) else str(pred_idx),
        "confidence": confidence
    })

if __name__ == "__main__":
    # accessible from other devices on your LAN
    app.run(host="0.0.0.0", port=5000)
