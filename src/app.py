# G:\...\src\app.py

import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS
import joblib
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing import DataProcessingPipeline

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Load the entire pipeline object from the single .joblib file ---
pipeline_path = project_root / 'model' / 'pipeline.joblib'
pipeline = joblib.load(pipeline_path)
print("--- Full prediction pipeline loaded successfully ---")

# --- Define the API Prediction Endpoint ---
@app.route("/predict", methods=['POST'])
def predict():
    """Handles prediction requests by running the full pipeline."""
    try:
        data = request.get_json(force=True)
        raw_df = pd.DataFrame([data])
        
        prediction = pipeline.predict(raw_df)
        output = prediction[0]
        
        return jsonify({'sale_amount_prediction': round(output, 2)})
        
    except Exception as e:
        return jsonify({'error': str(e), 'type': type(e).__name__}), 500

# --- Health Check Endpoint (Good Practice) ---
@app.route("/", methods=['GET'])
def health_check():
    """Confirms the API is running."""
    return jsonify({"status": "API is running"}), 200