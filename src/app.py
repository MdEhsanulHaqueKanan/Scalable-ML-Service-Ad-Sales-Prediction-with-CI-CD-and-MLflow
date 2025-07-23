# G:\...\src\app.py

import pandas as pd
from flask import Flask, request, jsonify
import mlflow
from pathlib import Path

# Import the NEW master pipeline function
from data_processing import run_full_pipeline

# Initialize the Flask application
app = Flask(__name__)

# --- 1. Load Model and Training Columns ---
project_root = Path(__file__).resolve().parent.parent
mlruns_path = project_root / 'mlruns'
mlflow.set_tracking_uri(mlruns_path.as_uri())

# !! IMPORTANT !!
# UPDATE THIS WITH THE RUN ID FROM YOUR LATEST MLFLOW RUN
RUN_ID = '1afc5ac32345429f8608f19ddeb793a9' # <--- USE YOUR LATEST RUN ID
logged_model_uri = f'runs:/{RUN_ID}/random_forest_model'
model = mlflow.pyfunc.load_model(logged_model_uri)

# Load the training columns from the model's signature
TRAINING_COLUMNS = model.metadata.get_input_schema().input_names()
print("Model and training columns loaded successfully!")

# --- 2. Define the Prediction Endpoint ---
@app.route("/predict", methods=['POST'])
def predict():
    """Endpoint to receive raw data and return a real prediction."""
    try:
        data = request.get_json(force=True)
        raw_df = pd.DataFrame([data])
        
        # Use the master pipeline function. It handles everything!
        processed_df = run_full_pipeline(raw_df, training_columns=TRAINING_COLUMNS)
        
        # Make a prediction
        prediction = model.predict(processed_df)
        output = prediction[0]
        
        return jsonify({'sale_amount_prediction': round(output, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# --- 3. Run the App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)