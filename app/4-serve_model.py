# 4-serve_model.py

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# === MODELLERİ YÜKLE ===
soh_model = joblib.load("model-2/models/RandomForest_discharge_duration.pkl")
soc_model = joblib.load("model-2/models/RandomForest_soc.pkl")

# === Ortak özellik isimleri (sıralı ve aynı olmalı) ===
features = joblib.load("model-2/models/features_order.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_values = data["features"]

        # Feature isimleri doğru sırayla burada olmalı:
        feature_names = ["cycle", "t_mean_IC5", "battery"]

        input_df = pd.DataFrame([input_values], columns=feature_names)

        predicted_soh = soh_model.predict(input_df)[0]
        predicted_soc = soc_model.predict(input_df)[0]

        return jsonify({
            "predicted_soh": float(predicted_soh),
            "predicted_soc": float(predicted_soc)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
