# /api/predict.py
import pandas as pd
from flask import Flask, request, jsonify
import joblib
from pathlib import Path

app = Flask(__name__)

# --- Load the entire pipeline object from the single .joblib file ---
# This path is relative to the project root, which Vercel understands
pipeline_path = Path(__file__).resolve().parent.parent / 'model' / 'pipeline.joblib'
pipeline = joblib.load(pipeline_path)
print("--- Full prediction pipeline loaded successfully ---")

# Vercel automatically routes requests to /api/predict to this file.
# The Flask app object is used by Vercel's Python WSGI server.
# We only need to define the endpoint we care about.
@app.route('/api/predict', methods=['POST'])
def predict_handler():
    """Handles prediction requests by running the full pipeline."""
    try:
        # Get raw JSON data
        data = request.get_json(force=True)
        # Convert to DataFrame. Note: Column names must match the original raw data.
        raw_df = pd.DataFrame([data])
        
        # The pipeline handles EVERYTHING: cleaning, feature engineering, and prediction
        prediction = pipeline.predict(raw_df)
        
        # The output of a scikit-learn regressor is a numpy array, get the first element
        output = prediction[0]
        
        return jsonify({'sale_amount_prediction': round(output, 2)})
        
    except Exception as e:
        # Provide a detailed error for debugging
        return jsonify({'error': str(e), 'type': type(e).__name__}), 500