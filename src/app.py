# G:\...\src\app.py

import pandas as pd
from flask import Flask, request, jsonify
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

# Import the master pipeline function
from .data_processing import run_full_pipeline

# Initialize the Flask application
app = Flask(__name__)

# --- 1. Load Model from Model Registry (The ULTIMATE Production Way) ---
# By not setting a tracking_uri, MLflow will default to the correct relative path
# which becomes /app/mlruns inside the container.
client = MlflowClient()

# Define the model name and the alias we want to load
model_name = "ad-sales-regressor"
model_alias = "production"

# Use the client to get the specific version URI for the alias
model_version_details = client.get_model_version_by_alias(name=model_name, alias=model_alias)
model_uri = model_version_details.source

# Load the production model using its direct source URI
model = mlflow.pyfunc.load_model(model_uri)

# Load the training columns from the model's signature
TRAINING_COLUMNS = model.metadata.get_input_schema().input_names()

print(f"--- Model Loaded Successfully ---")
print(f"Model Name: {model_name}")
print(f"Version: {model_version_details.version}")
print(f"Alias: {model_alias}")
print("---------------------------------")


# --- 2. Define the Prediction Endpoint ---
@app.route("/predict", methods=['POST'])
def predict():
    """Endpoint to receive raw data, process it, and return a real prediction."""
    try:
        # Get raw JSON data from the POST request
        data = request.get_json(force=True)
        
        # THIS IS THE FIX: Standardize keys to lowercase before creating the DataFrame
        data_lower = {key.lower(): value for key, value in data.items()}
        raw_df = pd.DataFrame([data_lower])
        
        # Process the now-standardized raw data using our complete pipeline
        processed_df = run_full_pipeline(raw_df, training_columns=TRAINING_COLUMNS)
        
        # Make a prediction
        prediction = model.predict(processed_df)
        output = prediction[0]
        
        # Return the final prediction
        return jsonify({'sale_amount_prediction': round(output, 2)})

    except Exception as e:
        # Return a detailed error if anything goes wrong
        return jsonify({'error': str(e), 'type': type(e).__name__}), 400

# --- 3. Run the App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)