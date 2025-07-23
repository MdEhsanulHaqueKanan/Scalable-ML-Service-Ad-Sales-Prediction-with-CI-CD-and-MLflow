# G:\...\tests\test_data_processing.py

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from pandas.testing import assert_frame_equal

# This ensures the test file can find our 'src' module
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import all the functions we want to test
from src.data_processing import (
    load_and_standardize_columns,
    clean_monetary_columns,
    standardize_date_column,
    clean_categorical_columns,
    handle_missing_values
)

@pytest.fixture
def raw_data_path() -> Path:
    """Fixture to provide the path to the raw data file."""
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    return project_root / 'data' / 'GoogleAds_DataAnalytics_Sales_Uncleaned.csv'

# --- Test 1: Column Standardization ---
def test_load_and_standardize_columns(raw_data_path):
    """
    GIVEN a valid path to the raw data CSV.
    WHEN the load_and_standardize_columns function is called.
    THEN the returned DataFrame has correctly formatted column names.
    """
    df = load_and_standardize_columns(raw_data_path)
    expected_cols = [
        'ad_id', 'campaign_name', 'clicks', 'impressions', 'cost', 'leads', 
        'conversions', 'conversion_rate', 'sale_amount', 'ad_date', 'location', 
        'device', 'keyword'
    ]
    assert df.columns.tolist() == expected_cols

# --- Test 2: Monetary Cleaning ---
def test_clean_monetary_columns():
    """
    GIVEN a DataFrame with messy monetary strings.
    WHEN the clean_monetary_columns function is called.
    THEN the monetary columns are converted to floats, and invalid values become NaN.
    """
    # GIVEN
    dirty_df = pd.DataFrame({
        'cost': ['$100.50', '₹50', 'invalid', np.nan],
        'sale_amount': ['$1,000.00', '250.75', '$300', 'error']
    })
    
    # WHEN
    cleaned_df = clean_monetary_columns(dirty_df)
    
    # THEN
    assert cleaned_df['cost'].dtype == 'float64'
    assert cleaned_df['sale_amount'].dtype == 'float64'
    assert pd.isna(cleaned_df.loc[2, 'cost']) # Check that 'invalid' became NaN
    assert pd.isna(cleaned_df.loc[3, 'sale_amount']) # Check that 'error' became NaN
    assert cleaned_df.loc[0, 'cost'] == 100.50

# --- Test 3: Date Standardization ---
def test_standardize_date_column():
    """
    GIVEN a DataFrame with mixed date string formats.
    WHEN the standardize_date_column function is called.
    THEN the date column is converted to datetime objects, and invalid dates become NaT.
    """
    # GIVEN
    dirty_df = pd.DataFrame({
        'ad_date': ['2024-01-15', '16-01-2024', '2024/01/17', 'not a date']
    })
    
    # WHEN
    cleaned_df = standardize_date_column(dirty_df)
    
    # THEN
    assert cleaned_df['ad_date'].dtype == 'datetime64[ns]'
    assert cleaned_df.loc[0, 'ad_date'] == pd.Timestamp('2024-01-15')
    assert cleaned_df.loc[1, 'ad_date'] == pd.Timestamp('2024-01-16')
    assert pd.isna(cleaned_df.loc[3, 'ad_date']) # Check that 'not a date' became NaT

# --- Test 4: Categorical Cleaning ---
def test_clean_categorical_columns():
    """
    GIVEN a DataFrame with inconsistent casing and typos in categorical columns.
    WHEN the clean_categorical_columns function is called.
    THEN the columns are lowercased and typos are corrected.
    """
    # GIVEN
    dirty_df = pd.DataFrame({
        'campaign_name': ['Data Anlytics Corse', 'dataanalyticscourse'],
        'location': ['HYDERABAD', 'hydrebad'],
        'device': ['Desktop', 'mobile'],
        'keyword': ['data anaytics training', 'learn data analytics']
    })
    
    # WHEN
    cleaned_df = clean_categorical_columns(dirty_df)
    
    # THEN
    assert cleaned_df.loc[0, 'campaign_name'] == 'data analytics course'
    assert cleaned_df.loc[1, 'campaign_name'] == 'data analytics course'
    assert cleaned_df.loc[1, 'location'] == 'hyderabad'
    assert cleaned_df.loc[0, 'device'] == 'desktop'

# --- Test 5: Missing Value Handling ---
def test_handle_missing_values():
    """
    GIVEN a DataFrame with missing values and a 'conversion_rate' column.
    WHEN the handle_missing_values function is called.
    THEN the 'conversion_rate' column is dropped and NaNs in other specified columns are filled with the median.
    """
    # GIVEN
    dirty_df = pd.DataFrame({
        'clicks': [10, 20, np.nan],
        'impressions': [100, np.nan, 300],
        'cost': [5, 10, 15],
        'leads': [1, 2, 3],
        'conversions': [1, np.nan, 3],
        'sale_amount': [np.nan, 50, 100],
        'conversion_rate': [0.1, 0.2, 0.3]
    })
    
    # WHEN
    cleaned_df = handle_missing_values(dirty_df)
    
    # THEN
    assert 'conversion_rate' not in cleaned_df.columns
    assert cleaned_df['clicks'].isnull().sum() == 0
    assert cleaned_df.loc[2, 'clicks'] == 15.0 # Median of [10, 20] is 15
    assert cleaned_df['sale_amount'].isnull().sum() == 0
    assert cleaned_df.loc[0, 'sale_amount'] == 75.0 # Median of [50, 100] is 75