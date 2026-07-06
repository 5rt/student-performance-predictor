"""Flask web app. Run it, then open http://localhost:5000"""
import json
import os
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
MODEL_PATH = os.path.join("model", "model.pkl")
METRICS_PATH = os.path.join("model", "metrics.json")
PASS_THRESHOLD = 50.0

bundle = joblib.load(MODEL_PATH)
MODEL = bundle["model"]
FEATURES = bundle["features"]

METRICS = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH) as f:
        METRICS = json.load(f)


def predict_score(values):
    row = pd.DataFrame([{f: float(values[f]) for f in FEATURES}])
    score = float(MODEL.predict(row)[0])
    score = max(0.0, min(100.0, score))
    return {
        "predicted_score": round(score, 1),
        "passed": score >= PASS_THRESHOLD,
        "model_used": METRICS.get("best_model", "model"),
    }


@app.route("/")
def index():
    return render_template("index.html", metrics=METRICS)


@app.route("/predict", methods=["POST"])
def predict_form():
    try:
        values = {f: request.form[f] for f in FEATURES}
        result = predict_score(values)
    except (KeyError, ValueError):
        return render_template("index.html", metrics=METRICS,
                               error="Please fill in every field with a number.")
    return render_template("index.html", metrics=METRICS,
                           result=result, submitted=values)


@app.route("/api/predict", methods=["POST"])
def predict_api():
    data = request.get_json(silent=True) or {}
    try:
        values = {f: data[f] for f in FEATURES}
        return jsonify(predict_score(values))
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Provide numbers for all fields.",
                        "required_fields": FEATURES}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)