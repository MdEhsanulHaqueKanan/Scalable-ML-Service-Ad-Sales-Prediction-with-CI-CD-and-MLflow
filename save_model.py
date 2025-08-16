# G:\...\save_model.py
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import joblib
import shutil

# Import our custom transformer class
from src.data_processing import DataProcessingPipeline

print("--- Starting full pipeline saving process ---")

# 1. Define paths
project_root = Path(__file__).resolve().parent
data_path = project_root / 'data' / 'GoogleAds_DataAnalytics_Sales_Uncleaned.csv'
output_model_path = project_root / 'model'

# 2. Clean up old model folder
if output_model_path.exists():
    shutil.rmtree(output_model_path)
output_model_path.mkdir(parents=True, exist_ok=True)

# 3. Load raw data
raw_df = pd.read_csv(data_path)
# Prepare target variable y (must be cleaned separately as it's not part of the pipeline's X)
y = pd.to_numeric(raw_df['Sale_Amount'].astype(str).str.replace(r'[$,₹]', '', regex=True), errors='coerce').fillna(0)
# Use all other columns as features X
X = raw_df.drop('Sale_Amount', axis=1, errors='ignore')


# 4. Define the full pipeline
# This chains our custom data processor with the model
full_pipeline = Pipeline(steps=[
    ('preprocessor', DataProcessingPipeline()),
    ('regressor', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
])

# 5. Train the entire pipeline on the full dataset
full_pipeline.fit(X, y)
print("Full scikit-learn pipeline trained successfully.")

# 6. Save the entire pipeline object to a single file
pipeline_file_path = output_model_path / 'pipeline.joblib'
joblib.dump(full_pipeline, pipeline_file_path)

print(f"\n--- SUCCESS! ---")
print(f"Full pipeline has been saved to: {pipeline_file_path}")