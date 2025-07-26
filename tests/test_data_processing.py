# G:\...\tests\test_data_processing.py
import pandas as pd
from src.data_processing import run_full_pipeline

def test_pipeline_handles_messy_data_and_encodes():
    # GIVEN: A dictionary with messy, non-standardized column names and values
    raw_data = {
        "Campaign_Name ": "Data Analytcis Course ", # With space
        "Clicks": 150,
        "Impressions": 5000,
        "Cost": "$200",
        "Ad_Date": "25-07-2025",
        "Location": "hydrebad",
        "Device": "DESKTOP"
    }
    raw_df = pd.DataFrame([raw_data])

    # WHEN
    processed_df = run_full_pipeline(raw_df)

    # THEN
    # Check if a column created by one-hot encoding exists
    assert 'device_desktop' in processed_df.columns
    # Check if the date features were created
    assert 'day_of_week' in processed_df.columns
    # Check if the cost was converted to a number
    assert processed_df['cost'].dtype == 'float64'