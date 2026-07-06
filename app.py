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

# Allowed range for each input. Server-side validation is the real safeguard —
# the form's min/max only guides the browser and can be bypassed.
VALIDATION = {
    "hours_studied":     (0, 24),
    "attendance":        (0, 100),
    "previous_score":    (0, 100),
    "sleep_hours":       (0, 24),
    "tutoring_sessions": (0, 50),
}

bundle = joblib.load(MODEL_PATH)
MODEL = bundle["model"]
FEATURES = bundle["features"]

METRICS = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH) as f:
        METRICS = json.load(f)


class ValidationError(Exception):
    """Raised when an input is missing, non-numeric, or out of range."""


def validate_and_parse(values):
    """Check every feature is present, numeric, and within range."""
    parsed = {}
    for f in FEATURES:
        raw = values.get(f)
        if raw is None or raw == "":
            raise ValidationError(f"Missing value for '{f}'.")
        try:
            num = float(raw)
        except (TypeError, ValueError):
            raise ValidationError(f"'{f}' must be a number.")
        lo, hi = VALIDATION[f]
        if not (lo <= num <= hi):
            raise ValidationError(f"'{f}' must be between {lo} and {hi} (got {num}).")
        parsed[f] = num
    return parsed


def predict_score(parsed):
    row = pd.DataFrame([parsed])
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
        parsed = validate_and_parse(request.form)
        result = predict_score(parsed)
    except ValidationError as e:
        return render_template("index.html", metrics=METRICS, error=str(e))
    return render_template("index.html", metrics=METRICS,
                           result=result, submitted=request.form)


@app.route("/api/predict", methods=["POST"])
def predict_api():
    data = request.get_json(silent=True) or {}
    try:
        parsed = validate_and_parse(data)
    except ValidationError as e:
        return jsonify({"error": str(e), "required_fields": FEATURES}), 400
    return jsonify(predict_score(parsed))


@app.route("/health")
def health():
    """Simple health check — used by deployment platforms and monitoring."""
    return jsonify({"status": "ok", "model": METRICS.get("best_model", "unknown")})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)