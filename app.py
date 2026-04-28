from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "Climate Risk API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    features = np.array([[
        data['co2_emissions'],
        data['temperature'],
        data['renewable_energy'],
        data['population']
    ]])
    
    prediction = model.predict(features)[0]
    
    return jsonify({"risk_level": prediction})

if __name__ == "__main__":
    app.run(debug=True)