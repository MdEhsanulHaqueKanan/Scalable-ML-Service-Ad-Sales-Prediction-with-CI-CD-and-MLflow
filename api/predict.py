# /api/predict.py

import pandas as pd
from flask import Flask, request, jsonify
import mlflow.pyfunc
from pathlib import Path

# NOTE: We need to import our processing function.
# To do this in the Vercel environment, we need to add the project root to the path.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_processing import run_full_pipeline

# Initialize a Flask app. Vercel will handle running it.
app = Flask(__name__)

# --- Load the model ONCE when the serverless function starts up ---
# The model is baked into the deployment, so we use a relative path.
model_path = Path(__file__).resolve().parent.parent / 'model' / 'random_forest_model'
model = mlflow.pyfunc.load_model(model_path)
TRAINING_COLUMNS = model.metadata.get_input_schema().input_names()
print("--- Model Loaded Successfully ---")


# --- Define the API endpoint ---
# Vercel will automatically route requests from /api/predict to this function.
@app.route('/api/predict', methods=['POST'])
def predict_handler():
    """Handles prediction requests."""
    try:
        data = request.get_json(force=True)
        data_lower = {key.lower(): value for key, value in data.items()}
        raw_df = pd.DataFrame([data_lower])
        
        processed_df = run_full_pipeline(raw_df, training_columns=TRAINING_COLUMNS)
        
        prediction = model.predict(processed_df)
        output = prediction[0]
        
        return jsonify({'sale_amount_prediction': round(output, 2)})

    except Exception as e:
        return jsonify({'error': str(e), 'type': type(e).__name__}), 500