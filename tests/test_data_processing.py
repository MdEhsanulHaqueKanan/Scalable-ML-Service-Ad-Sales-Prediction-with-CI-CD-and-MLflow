# G:\...\tests\test_data_processing.py
import pandas as pd
from src.data_processing import run_full_pipeline

def test_pipeline_creates_dummified_columns():
    """
    GIVEN a DataFrame with multiple messy rows.
    WHEN the full pipeline is run.
    THEN the output should be clean, processed, and correctly one-hot encoded.
    """
    # GIVEN: A DataFrame with multiple rows to ensure dummies are created correctly
    raw_data = [
        {"Campaign_Name": "Data Course ", "Device": "DESKTOP", "Location": "hydrebad", "Ad_Date": "2025-01-01", "Cost": "$100"},
        {"Campaign_Name": "Data Course", "Device": "mobile", "Location": "hyderabad", "Ad_Date": "2025-01-02", "Cost": "$150"}
    ]
    raw_df = pd.DataFrame(raw_data)

    # WHEN
    processed_df = run_full_pipeline(raw_df)

    # THEN
    # Check if the dummified column for 'mobile' was created (since 'desktop' might be dropped)
    assert 'device_mobile' in processed_df.columns
    # Check that the original 'device' column is gone
    assert 'device' not in processed_df.columns
    # Check that date features were created
    assert 'day_of_week' in processed_df.columns
    # Check that cost was converted
    assert processed_df['cost'].dtype == 'float64'