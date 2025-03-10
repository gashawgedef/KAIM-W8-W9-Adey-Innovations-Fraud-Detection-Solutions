from flask import Flask, request, jsonify
import pickle
import pandas as pd
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(filename="api.log", level=logging.INFO)

# Load model
with open("models/fraud_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame([data])
        
        # Ensure feature order matches training
        feature_order = model.feature_names_in_
        df = df[feature_order]
        
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        logging.info(f"Prediction made: {prediction}, Probability: {probability}")
        return jsonify({"prediction": int(prediction), "probability": float(probability)})
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)