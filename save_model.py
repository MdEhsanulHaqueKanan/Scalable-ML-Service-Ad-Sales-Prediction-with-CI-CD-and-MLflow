# save_model.py

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import mlflow.sklearn
import shutil

# Import our pipeline
from src.data_processing import run_full_pipeline

print("--- Starting model training and saving process ---")

# 1. Define paths
project_root = Path(__file__).resolve().parent
data_path = project_root / 'data' / 'GoogleAds_DataAnalytics_Sales_Uncleaned.csv'
output_model_path = project_root / 'model'

# 2. Clean up old model folder if it exists
if output_model_path.exists():
    print(f"Removing existing model folder: {output_model_path}")
    shutil.rmtree(output_model_path)

print(f"Creating new model folder: {output_model_path}")
output_model_path.mkdir(parents=True, exist_ok=True)

# 3. Load and process data
raw_df = pd.read_csv(data_path)
df_model_ready = run_full_pipeline(raw_df)
print("Data loaded and processed.")

# 4. Split data
X = df_model_ready.drop('sale_amount', axis=1, errors='ignore')
y = df_model_ready['sale_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split.")

# 5. Train model
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
print("Model trained.")

# 6. Save the model using MLflow's format to the desired directory
mlflow.sklearn.save_model(
    sk_model=model,
    path=output_model_path / "random_forest_model", # Save directly to model/random_forest_model
    input_example=X_train.head(5),
    signature=mlflow.models.infer_signature(X_train.head(5), model.predict(X_train.head(5)))
)

print("\n--- SUCCESS! ---")
print(f"Model has been saved to: {output_model_path / 'random_forest_model'}")