# backend/app.py
import os, json
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from joblib import load

app = Flask(__name__)
CORS(app)

# load config
CFG_PATH = Path(__file__).parent / "config.json"
CONFIG = json.loads(CFG_PATH.read_text())

# load models from ../models/
MODEL_DIR = Path(__file__).parent.parent / "models"
MODELS = {}
for disease in CONFIG.keys():
    p = MODEL_DIR / f"{disease}_model.pkl"
    if p.exists():
        MODELS[disease] = load(p)
        print("Loaded model:", p)
    else:
        MODELS[disease] = None
        print("Model missing:", p)

def get_fields(disease):
    csv_path = Path(__file__).parent.parent / CONFIG[disease]["file"]
    df = pd.read_csv(csv_path)
    target = CONFIG[disease]["target"]
    # Exclude target and ID columns
    fields = [c for c in df.columns if c != target and c.lower() not in ['id', 'index']]
    return fields

@app.route("/api/diseases", methods=["GET"])
def diseases():
    result = {}
    for d in CONFIG.keys():
        try:
            result[d] = {"fields": get_fields(d)}
        except Exception as e:
            result[d] = {"error": str(e)}
    return jsonify(result)

@app.route("/api/predict/<disease>", methods=["POST"])
def predict(disease):
    if disease not in MODELS or MODELS[disease] is None:
        return jsonify({"error":"model not available"}), 404

    payload = request.get_json()
    if not isinstance(payload, dict):
        return jsonify({"error":"expected JSON object in body"}), 400

    fields = get_fields(disease)
    row = {k: payload.get(k, None) for k in fields}
    df = pd.DataFrame([row])
    # convert numeric-like columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    model = MODELS[disease]
    try:
        prob = float(model.predict_proba(df)[:,1][0])
        pred = int(model.predict(df)[0])
    except Exception as e:
        return jsonify({"error": f"model prediction failed: {e}"}), 500

    return jsonify({"disease": disease, "prediction": pred, "probability": prob})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
