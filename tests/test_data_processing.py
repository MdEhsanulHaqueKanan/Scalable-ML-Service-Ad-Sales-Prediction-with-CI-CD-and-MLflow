import pandas as pd
import numpy as np
import pytest
from pathlib import Path

# This is now a standard library import because of our setup.py
from src.data_processing import run_full_pipeline

def test_run_full_pipeline_from_dict():
    """
    GIVEN a dictionary representing a raw data row.
    WHEN the run_full_pipeline function is called.
    THEN it returns a processed DataFrame with the correct shape and no NaNs.
    """
    # GIVEN: A sample of raw, messy data as a dictionary
    raw_data = {
        "Campaign_Name": "Data Analytcis Course",
        "Clicks": 150,
        "Impressions": 5000,
        "Cost": "$200",
        "Leads": 20,
        "Conversions": 10,
        "Ad_Date": "2025-07-24",
        "Location": "hydrebad",
        "Device": "DESKTOP",
        "Keyword": "data analitics online"
    }
    raw_df = pd.DataFrame([raw_data])

    # WHEN
    processed_df = run_full_pipeline(raw_df)

    # THEN: Check for expected outcomes
    # 1. The output should be a pandas DataFrame
    assert isinstance(processed_df, pd.DataFrame)
    
    # 2. There should be no missing values in the final output
    assert processed_df.isnull().sum().sum() == 0
    
    # 3. Check if one-hot encoding was successful by looking for an expected column
    assert 'device_desktop' in processed_df.columns or 'device_mobile' in processed_df.columns