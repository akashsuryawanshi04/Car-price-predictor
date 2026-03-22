"""
app.py
======
Flask web application for the Car Price Predictor.
Serves the UI and exposes a /predict REST endpoint.

Run:
    python app.py
    Then open http://127.0.0.1:5000

Author: Your Name
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "car_price_model.pkl")
META_PATH  = os.path.join(BASE_DIR, "..", "model", "model_meta.json")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Load model & metadata at startup ─────────────────────────────────────────
print("[STARTUP] Loading model …")
model = joblib.load(MODEL_PATH)

with open(META_PATH) as f:
    meta = json.load(f)

print("[STARTUP] ✅ Model and metadata loaded.")
print(f"          R²  = {meta['metrics']['R2']}")
print(f"          MAE = ₹ {meta['metrics']['MAE']:,.0f}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html", meta=meta)


@app.route("/api/meta")
def api_meta():
    """Return model metadata (used by the front-end to populate dropdowns)."""
    return jsonify(meta)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    POST  /api/predict
    Body (JSON):
        {
          "name":      "Maruti Suzuki Alto",
          "company":   "Maruti",
          "year":      2016,
          "kms_driven": 35000,
          "fuel_type": "Petrol"
        }

    Returns:
        { "predicted_price": 215000, "formatted": "₹ 2,15,000" }
    """
    try:
        data = request.get_json(force=True)

        required = ["name", "company", "year", "kms_driven", "fuel_type"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Build a single-row DataFrame matching training feature order
        input_df = pd.DataFrame([{
            "name":       str(data["name"]),
            "company":    str(data["company"]),
            "fuel_type":  str(data["fuel_type"]),
            "year":       int(data["year"]),
            "kms_driven": int(data["kms_driven"]),
        }])

        price = float(model.predict(input_df)[0])
        price = max(price, 10_000)   # floor sanity check

        # Indian number format helper
        def indian_format(n):
            n = int(round(n))
            s = str(n)
            if len(s) <= 3:
                return s
            last3 = s[-3:]
            rest  = s[:-3]
            groups = []
            while len(rest) > 2:
                groups.append(rest[-2:])
                rest = rest[:-2]
            if rest:
                groups.append(rest)
            return ",".join(reversed(groups)) + "," + last3

        formatted = f"₹ {indian_format(price)}"

        return jsonify({
            "predicted_price": round(price),
            "formatted":       formatted,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model": "LinearRegression", "r2": meta["metrics"]["R2"]})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
