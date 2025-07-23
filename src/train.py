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

# Import the NEW master pipeline function from our data processing script
from data_processing import run_full_pipeline

if __name__ == '__main__':
    # --- 1. Setup Paths and MLflow ---
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / 'data' / 'GoogleAds_DataAnalytics_Sales_Uncleaned.csv'
    mlruns_path = project_root / 'mlruns'
    
    mlflow.set_tracking_uri(mlruns_path.as_uri())
    mlflow.set_experiment("Ad Sales Prediction")

    with mlflow.start_run():
        # --- 2. Load Raw Data and Process It ---
        # Load the raw data
        raw_df = pd.read_csv(data_path)
        # Process the raw data using our complete, imported pipeline
        df_model_ready = run_full_pipeline(raw_df)

        print("Step 1 (Training): Data loaded and processed via master pipeline.")

        # --- 3. Split Data ---
        X = df_model_ready.drop('sale_amount', axis=1)
        y = df_model_ready['sale_amount']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Step 2 (Training): Data split into training and testing sets.")

        # --- 4. Train Model ---
        n_estimators = 100
        max_depth = 10
        random_state = 42

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)
        print("Step 3 (Training): Model training completed.")

        # --- 5. Evaluate Model ---
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        
        print(f"\nModel Performance:")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R² Score: {r2:.2f}")

        # --- 6. Log Model with Signature ---
        input_example = X_train.head(5)
        signature = infer_signature(input_example, model.predict(input_example))

        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="random_forest_model",
            signature=signature,
            input_example=input_example
        )
        
        print("\nMLflow run completed with model signature.")
        print(f"To view the UI, run 'mlflow ui' in your terminal from the project root.")