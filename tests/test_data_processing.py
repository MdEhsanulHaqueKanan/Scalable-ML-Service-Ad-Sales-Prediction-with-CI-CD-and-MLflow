# G:\...\tests\test_data_processing.py
import pandas as pd
from src.data_processing import run_full_pipeline

def test_pipeline_creates_dummified_columns():
    raw_data = {
        "Campaign_Name": "Data Analytcis Course",
        "Clicks": 150, "Impressions": 5000, "Cost": "$200", "Leads": 20,
        "Conversions": 10, "Ad_Date": "2025-07-24", "Location": "hydrebad",
        "Device": "DESKTOP", "Keyword": "data analitics online"
    }
    raw_df = pd.DataFrame([raw_data])
    processed_df = run_full_pipeline(raw_df)
    assert 'device_desktop' in processed_df.columns