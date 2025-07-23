# G:\...\src\train.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature # Import the signature tool

# Import the data processing pipeline we built
from data_processing import (
    load_and_standardize_columns,
    clean_monetary_columns,
    standardize_date_column,
    clean_categorical_columns,
    handle_missing_values
)

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # (This function is unchanged)
    df_eng = df.copy()
    df_eng['day_of_week'] = df_eng['ad_date'].dt.dayofweek
    df_eng['month'] = df_eng['ad_date'].dt.month
    df_eng['day_of_month'] = df_eng['ad_date'].dt.day
    df_eng = df_eng.drop(columns=['ad_date', 'ad_id'])
    categorical_cols = ['campaign_name', 'location', 'device', 'keyword']
    df_eng = pd.get_dummies(df_eng, columns=categorical_cols, drop_first=True)
    print("Step 6: Feature engineering completed.")
    return df_eng

if __name__ == '__main__':
    # --- 1. Define Paths Robustly ---
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    data_path = project_root / 'data' / 'GoogleAds_DataAnalytics_Sales_Uncleaned.csv'
    mlruns_path = project_root / 'mlruns'

    # --- 2. Setup MLflow ---
    mlflow.set_tracking_uri(mlruns_path.as_uri())
    mlflow.set_experiment("Ad Sales Prediction")

    with mlflow.start_run():
        # --- 3. Load and Process Data ---
        df1 = load_and_standardize_columns(data_path)
        df2 = clean_monetary_columns(df1)
        df3 = standardize_date_column(df2)
        df4 = clean_categorical_columns(df3)
        df_clean = handle_missing_values(df4)
        df_model_ready = feature_engineering(df_clean)

        # --- 4. Split Data ---
        X = df_model_ready.drop('sale_amount', axis=1)
        y = df_model_ready['sale_amount']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # --- 5. Train Model ---
        n_estimators = 100
        max_depth = 10
        random_state = 42

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)

        # --- 6. Evaluate Model ---
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        
        print(f"\nModel Performance:")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R² Score: {r2:.2f}")

        # --- 7. Log Model with Signature (The 10/10 way) ---
        # Create an input example from our training data
        input_example = X_train.head(5)
        # Infer the signature from the input example and model predictions
        signature = infer_signature(input_example, model.predict(input_example))

        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="random_forest_model", # This is the modern argument name
            signature=signature,
            input_example=input_example
        )
        
        print("\nMLflow run completed with model signature.")
        print(f"To view the UI, run 'mlflow ui' in your terminal from the project root.")