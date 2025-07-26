# G:\...\tests\test_data_processing.py
import pandas as pd
from src.data_processing import run_full_pipeline

def test_pipeline_handles_messy_data_and_encodes():
    # GIVEN
    raw_data = [
        {"Campaign_Name": " Data Course ", "Device": "DESKTOP", "Cost": "$100.50"}, # Added decimal
        {"Campaign_Name": "Data Course", "Device": "mobile", "Cost": "150"}
    ]
    raw_df = pd.DataFrame(raw_data)

    # WHEN
    processed_df = run_full_pipeline(raw_df)

    # THEN
    assert 'device_mobile' in processed_df.columns
    assert processed_df['cost'].dtype == 'float64'