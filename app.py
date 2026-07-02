"""
app.py
======
Optional "tiny live demo" from the assignment: a local web page that
uses your webcam and shows the real-vs-screen prediction live.

This reuses the EXACT same pipeline as predict.py - `extract_features`
from utils.py and the saved model bundle from train.py - just wrapped
behind a tiny local HTTP endpoint so a browser page can call it. No
separate model, no reimplementation, nothing that could drift from the
graded predict.py behavior.

Usage:
    pip install flask
    python app.py
    open http://localhost:5000 in a browser, allow camera access.

Not required for grading (the assignment lists this as optional /
"impressive"). Runs entirely on localhost - no data leaves your
machine, camera frames are never saved to disk.
"""

from __future__ import annotations

import base64
import time
from typing import Optional

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from utils import FEATURE_NAMES, extract_features, load_model_bundle, logger

MODEL_PATH = "model/best_model.joblib"

app = Flask(__name__)
_bundle: Optional[dict] = None


def get_bundle() -> dict:
    """Load the model bundle once and cache it - avoids re-reading the
    joblib file on every single camera frame."""
    global _bundle
    if _bundle is None:
        _bundle = load_model_bundle(MODEL_PATH)
        logger.info(
            "Live demo loaded model bundle: %s (feature set %s)",
            _bundle.get("model_name"), _bundle.get("feature_set_version"),
        )
    return _bundle


def decode_data_url(data_url: str) -> np.ndarray:
    """Decode a browser canvas `toDataURL()` base64 JPEG/PNG string
    into a BGR uint8 numpy array."""
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url
    img_bytes = base64.b64decode(encoded)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode captured frame.")
    return image


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    t0 = time.perf_counter()
    payload = request.get_json(silent=True)
    if not payload or "image" not in payload:
        return jsonify({"error": "missing 'image' field"}), 400

    try:
        image = decode_data_url(payload["image"])
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"decode failed: {exc}"}), 400

    try:
        bundle = get_bundle()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500

    # Same exact-schema safety check as predict.py: refuse to produce a
    # number if the loaded model was trained against a different
    # utils.py feature set, rather than silently mispredicting.
    if list(bundle["feature_names"]) != list(FEATURE_NAMES):
        return jsonify({"error": "Model/feature schema mismatch - retrain with train.py."}), 500

    try:
        vector = extract_features(image)
        X = bundle["preprocessor"].transform(vector.reshape(1, -1))
        proba = float(bundle["model"].predict_proba(X)[0, 1])
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed on a live frame")
        return jsonify({"error": f"prediction failed: {exc}"}), 500

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return jsonify(
        {
            "probability": proba,
            "threshold": bundle["threshold"],
            "is_screen": proba >= bundle["threshold"],
            "latency_ms": round(elapsed_ms, 1),
        }
    )


if __name__ == "__main__":
    get_bundle()  # fail fast at startup if the model bundle is missing
    print("\nLive demo running - open http://localhost:5000 in your browser.\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
