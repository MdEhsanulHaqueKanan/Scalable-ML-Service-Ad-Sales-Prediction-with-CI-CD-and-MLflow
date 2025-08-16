# G:\...\save_model.py
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import joblib
import shutil

# Import our custom transformer class
from src.data_processing import run_full_pipeline

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
# Prepare target variable y (THIS IS THE CORRECTED LINE)
y = pd.to_numeric(raw_df['Sale_Amount'].astype(str).str.replace(r'[$,₹]', '', regex=True), errors='coerce').fillna(0)
# Process features X using the pipeline
X_processed = run_full_pipeline(raw_df.drop('Sale_Amount', axis=1, errors='ignore'))


# 4. Train the model
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_processed, y)
print("Model trained.")

# 5. Save the model and the final training columns
joblib.dump(model, output_model_path / 'model.joblib')
joblib.dump(list(X_processed.columns), output_model_path / 'columns.joblib')

print(f"\n--- SUCCESS! ---")
print(f"Model and columns saved to: {output_model_path}")