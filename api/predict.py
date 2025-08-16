# /api/predict.py
import pandas as pd
from flask import Flask, request, jsonify
import mlflow.pyfunc
from pathlib import Path
import sys

# Add the project root to the Python path so we can import 'src'
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing import run_full_pipeline

app = Flask(__name__)

# Load the model from the local /model directory
model_path = project_root / 'model' / 'random_forest_model'
model = mlflow.pyfunc.load_model(model_path)
TRAINING_COLUMNS = model.metadata.get_input_schema().input_names()
print("--- Model Loaded Successfully ---")

# This is the function Vercel will run
@app.route('/api/predict', methods=['POST'])
def predict_handler():
    try:
        data = request.get_json(force=True)
        raw_df = pd.DataFrame([data])
        processed_df = run_full_pipeline(raw_df, training_columns=TRAINING_COLUMNS)
        prediction = model.predict(processed_df)
        output = prediction[0]
        return jsonify({'sale_amount_prediction': round(output, 2)})
    except Exception as e:
        return jsonify({'error': str(e), 'type': type(e).__name__}), 500