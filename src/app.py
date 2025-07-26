# G:\...\src\app.py

import pandas as pd
from flask import Flask, request, jsonify, render_template
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

# Import the master pipeline function
from .data_processing import run_full_pipeline

# --- THIS IS THE FIX ---
# Define the paths to the templates and static folders relative to this script
template_folder_path = Path(__file__).resolve().parent.parent / 'templates'
static_folder_path = Path(__file__).resolve().parent.parent / 'static'

# Initialize the Flask application WITH EXPLICIT PATHS
app = Flask(__name__, template_folder=template_folder_path, static_folder=static_folder_path)

# --- The rest of the file is exactly the same ---

# --- 1. Load Model from Model Registry ---
client = MlflowClient()
model_name = "ad-sales-regressor"
model_alias = "production"
model_version_details = client.get_model_version_by_alias(name=model_name, alias=model_alias)
model_uri = model_version_details.source
model = mlflow.pyfunc.load_model(model_uri)
TRAINING_COLUMNS = model.metadata.get_input_schema().input_names()
print(f"--- Model Loaded Successfully ---")

# --- 2. Define the Home Page Endpoint ---
@app.route("/", methods=['GET'])
def home():
    """Renders the main HTML page with the input form."""
    return render_template('index.html')

# --- 3. Define the Prediction Endpoint ---
@app.route("/predict", methods=['POST'])
def predict():
    """Endpoint to receive raw data, process it, and return a real prediction."""
    try:
        data = request.get_json(force=True)
        data_lower = {key.lower(): value for key, value in data.items()}
        raw_df = pd.DataFrame([data_lower])
        processed_df = run_full_pipeline(raw_df, training_columns=TRAINING_COLUMNS)
        prediction = model.predict(processed_df)
        output = prediction[0]
        return jsonify({'sale_amount_prediction': round(output, 2)})
    except Exception as e:
        return jsonify({'error': str(e), 'type': type(e).__name__}), 400

# --- 4. Run the App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)