# G:\...\src\app.py

import pandas as pd
from flask import Flask, request, jsonify, render_template
from pathlib import Path
import mlflow.pyfunc # We only need the pyfunc module for loading

# Import the master pipeline function from our other script
from .data_processing import run_full_pipeline

# --- Define paths robustly to find the templates and static folders ---
template_folder_path = Path(__file__).resolve().parent.parent / 'templates'
static_folder_path = Path(__file__).resolve().parent.parent / 'static'

# Initialize the Flask application with explicit paths
app = Flask(__name__, template_folder=template_folder_path, static_folder=static_folder_path)


# --- 1. Load Model from the Local "model" Directory ---
# This path is relative to the project root, which will be /app inside the Docker container
model_uri = "model/random_forest_model"
model = mlflow.pyfunc.load_model(model_uri)

# Load the expected training columns from the model's signature
TRAINING_COLUMNS = model.metadata.get_input_schema().input_names()

print(f"--- Model Loaded Successfully from local path: {model_uri} ---")


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
        # Get raw JSON data from the POST request
        data = request.get_json(force=True)
        
        # Standardize keys to lowercase before creating the DataFrame
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

# --- 4. Run the App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)