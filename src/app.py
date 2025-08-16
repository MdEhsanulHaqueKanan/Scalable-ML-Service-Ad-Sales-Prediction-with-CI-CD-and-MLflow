# G:\...\src\app.py
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing import run_full_pipeline

app = Flask(__name__)
CORS(app)

model_path = project_root / 'model' / 'model.joblib'
columns_path = project_root / 'model' / 'columns.joblib'
model = joblib.load(model_path)
TRAINING_COLUMNS = joblib.load(columns_path)
print("--- Model and columns loaded successfully ---")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        raw_df = pd.DataFrame([data])
        processed_df = run_full_pipeline(raw_df, training_columns=TRAINING_COLUMNS)
        prediction = model.predict(processed_df)
        output = prediction[0]
        return jsonify({'sale_amount_prediction': round(output, 2)})
    except Exception as e:
        return jsonify({'error': str(e), 'type': type(e).__name__}), 500

@app.route("/", methods=['GET'])
def health_check():
    return jsonify({"status": "API is running"}), 200