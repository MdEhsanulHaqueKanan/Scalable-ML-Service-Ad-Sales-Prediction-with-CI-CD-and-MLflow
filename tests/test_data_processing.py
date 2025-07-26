import pandas as pd
import numpy as np
import pytest
from pathlib import Path

# This import works because of setup.py
from src.data_processing import run_full_pipeline

def test_run_full_pipeline_from_dict():
    """
    GIVEN a dictionary representing a raw data row with STANDARDIZED keys.
    WHEN the run_full_pipeline function is called.
    THEN it returns a processed DataFrame with the correct shape and no NaNs.
    """
    # GIVEN: Sample data with lowercase keys, just as the app will create.
    # The values can still be messy (e.g., "DESKTOP") to test the cleaning.
    raw_data = {
        "campaign_name": "Data Analytcis Course",
        "clicks": 150,
        "impressions": 5000,
        "cost": "$200",
        "leads": 20,
        "conversions": 10,
        "ad_date": "2025-07-24",
        "location": "hydrebad",
        "device": "DESKTOP",
        "keyword": "data analitics online"
    }
    raw_df = pd.DataFrame([raw_data])

    # WHEN
    processed_df = run_full_pipeline(raw_df)

    # THEN: Check for expected outcomes
    # 1. The output should be a pandas DataFrame
    assert isinstance(processed_df, pd.DataFrame)
    
    # 2. There should be no missing values in the final output
    assert processed_df.isnull().sum().sum() == 0
    
    # 3. Check if one-hot encoding was successful by looking for an expected column.
    #    This assertion will now pass.
    assert 'device_desktop' in processed_df.columns