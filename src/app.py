# G:\...\src\app.py

import pandas as pd
from flask import Flask, request, jsonify
import joblib
import json
from pathlib import Path

# Import the master pipeline function
from .data_processing import run_full_pipeline

# Initialize the Flask application
app = Flask(__name__)

# --- 1. Load Production Artifacts ---
# Define paths relative to the current script
base_dir = Path(__file__).resolve().parent.parent
model_path = base_dir / 'model' / 'model.joblib'
columns_path = base_dir / 'model' / 'columns.json'

# Load the model and the training columns
model = joblib.load(model_path)
with open(columns_path, 'r') as f:
    TRAINING_COLUMNS = json.load(f)
print("--- Model and training columns loaded successfully from local files ---")

# --- 2. Define the Prediction Endpoint ---
@app.route("/predict", methods=['POST'])
def predict():
    """Endpoint to receive raw data, process it, and return a real prediction."""
    try:
        data = request.get_json(force=True)
        raw_df = pd.DataFrame([data])
        
        processed_df = run_full_pipeline(raw_df, training_columns=TRAINING_COLUMNS)
        
        prediction = model.predict(processed_df)
        output = prediction[0]
        
        return jsonify({'sale_amount_prediction': round(output, 2)})

    except Exception as e:
        return jsonify({'error': str(e), 'type': type(e).__name__}), 400

# --- 3. Run the App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)