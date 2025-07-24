# G:\...\src\train.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import joblib # For saving the model
import json # For saving the column list

# Import the master pipeline function
from data_processing import run_full_pipeline

def train_model():
    """Main function to run the entire model training and logging pipeline."""
    # --- 1. Setup Paths ---
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / 'data' / 'GoogleAds_DataAnalytics_Sales_Uncleaned.csv'
    # Create a directory to save the final production model
    production_model_dir = project_root / 'model'
    production_model_dir.mkdir(exist_ok=True)

    # --- 2. MLflow Tracking ---
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Ad Sales Prediction")

    with mlflow.start_run() as run:
        # --- 3. Data Processing ---
        raw_df = pd.read_csv(data_path)
        df_model_ready = run_full_pipeline(raw_df)

        # --- 4. Data Splitting ---
        X = df_model_ready.drop('sale_amount', axis=1, errors='ignore')
        y = df_model_ready['sale_amount']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # --- 5. Model Training & Evaluation (same as before) ---
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        # --- 6. MLflow Logging (same as before) ---
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(sk_model=model, artifact_path="random_forest_model", signature=signature)
        
        # --- 7. SAVE PRODUCTION ARTIFACTS (The Critical New Step) ---
        # Save the trained model directly as a joblib file
        model_path = production_model_dir / 'model.joblib'
        joblib.dump(model, model_path)
        
        # Save the list of training columns as a json file
        columns_path = production_model_dir / 'columns.json'
        with open(columns_path, 'w') as f:
            json.dump(X_train.columns.tolist(), f)

        print(f"\n--- Production artifacts saved to '{production_model_dir}' ---")
        print(f"MLflow Run ID: {run.info.run_id}")

if __name__ == '__main__':
    train_model()