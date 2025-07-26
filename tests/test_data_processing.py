# G:\...\tests\test_data_processing.py

import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from src.data_processing import run_full_pipeline

def test_run_full_pipeline_from_dict():
    """
    GIVEN a dictionary representing raw data with messy, non-standardized keys.
    WHEN the run_full_pipeline function is called.
    THEN it returns a processed DataFrame with one-hot encoded columns.
    """
    # GIVEN: Data with TitleCase keys, just like the original CSV.
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

    # THEN
    # This assertion now correctly tests the end-to-end cleaning and feature engineering
    assert 'device_desktop' in processed_df.columns