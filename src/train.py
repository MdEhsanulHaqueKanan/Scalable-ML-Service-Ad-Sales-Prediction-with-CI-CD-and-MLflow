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
from data_processing import run_full_pipeline

def train_model():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / 'data' / 'GoogleAds_DataAnalytics_Sales_Uncleaned.csv'
    
    # REMOVED mlflow.set_tracking_uri - Let MLflow default to a local, relative ./mlruns
    mlflow.set_experiment("Ad Sales Prediction")

    with mlflow.start_run() as run: # Capture the run
        raw_df = pd.read_csv(data_path)
        df_model_ready = run_full_pipeline(raw_df)
        X = df_model_ready.drop('sale_amount', axis=1, errors='ignore')
        y = df_model_ready['sale_amount']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # ... (training and logging logic is the same) ...
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        input_example = X_train.head(5)
        signature = infer_signature(input_example, model.predict(input_example))
        mlflow.sklearn.log_model(sk_model=model, artifact_path="random_forest_model", signature=signature, input_example=input_example)
        
        print(f"\nMLflow Run ID: {run.info.run_id}")

if __name__ == '__main__':
    train_model()