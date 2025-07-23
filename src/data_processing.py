# G:\...\src\data_processing.py

import pandas as pd
import numpy as np
from pathlib import Path

def load_and_standardize_columns(file_path: str) -> pd.DataFrame:
    """
    Loads data and standardizes all column names.
    - Loads the data from a CSV file.
    - Standardizes column names (lowercase, strips whitespace, replaces spaces with underscores).
    
    Args:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The DataFrame with standardized column names.
    """
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    print("Step 1: Data loaded and column names standardized.")
    return df

def clean_monetary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans monetary columns by removing currency symbols and converting to numeric type.
    - Identifies columns to clean ('cost', 'sale_amount').
    - Removes '$' and '₹' symbols.
    - Converts columns to a numeric (float) data type. Errors are coerced to NaN.
    
    Args:
        df (pd.DataFrame): The DataFrame to process.
        
    Returns:
        pd.DataFrame: The DataFrame with cleaned monetary columns.
    """
    df_clean = df.copy()
    cols_to_clean = ['cost', 'sale_amount']
    
    for col in cols_to_clean:
        df_clean[col] = df_clean[col].astype(str).str.replace(r'[$,₹]', '', regex=True)
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
    print("Step 2: Monetary columns ('cost', 'sale_amount') cleaned.")
    return df_clean

def standardize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the 'ad_date' column to a consistent YYYY-MM-DD format.
    - Converts the 'ad_date' column to datetime objects, handling multiple formats.
    - Errors are coerced to NaT (Not a Time).
    
    Args:
        df (pd.DataFrame): The DataFrame to process.
        
    Returns:
        pd.DataFrame: The DataFrame with a standardized 'ad_date' column.
    """
    df_clean = df.copy()
    df_clean['ad_date'] = pd.to_datetime(df_clean['ad_date'], format='mixed', errors='coerce')
    print("Step 3: Date column ('ad_date') standardized.")
    return df_clean

def clean_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and standardizes categorical columns.
    - Converts specified columns to lowercase.
    - Corrects known typos and consolidates variations.
    
    Args:
        df (pd.DataFrame): The DataFrame to process.
        
    Returns:
        pd.DataFrame: The DataFrame with cleaned categorical columns.
    """
    df_clean = df.copy()
    
    cols_to_lower = ['campaign_name', 'location', 'device', 'keyword']
    for col in cols_to_lower:
        df_clean[col] = df_clean[col].str.lower()
        
    # Define mapping for corrections
    campaign_mapping = {
        'data anlytics corse': 'data analytics course',
        'data analytcis course': 'data analytics course',
        'data analytics corse': 'data analytics course',
        'dataanalyticscourse': 'data analytics course' # Added the final variation
    }
    location_mapping = { 'hyderbad': 'hyderabad', 'hydrebad': 'hyderabad' }
    keyword_mapping = {
        'data analitics online': 'data analytics online',
        'data anaytics training': 'data analytics training',
        'online data analytic': 'online data analytics'
    }
    
    df_clean['campaign_name'] = df_clean['campaign_name'].replace(campaign_mapping)
    df_clean['location'] = df_clean['location'].replace(location_mapping)
    df_clean['keyword'] = df_clean['keyword'].replace(keyword_mapping)
    
    print("Step 4: Categorical columns cleaned and standardized.")
    return df_clean

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values in the dataset.
    - Drops the 'conversion_rate' column due to excessive missing values.
    - Fills missing values in key numerical columns with their respective medians.
    
    Args:
        df (pd.DataFrame): The DataFrame to process.
        
    Returns:
        pd.DataFrame: The DataFrame with missing values handled.
    """
    df_clean = df.copy()
    
    # Drop column with too many missing values
    df_clean = df_clean.drop(columns=['conversion_rate'])
    
    # Columns to fill with median
    cols_to_fill = ['clicks', 'impressions', 'cost', 'leads', 'conversions', 'sale_amount']
    for col in cols_to_fill:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
        
    print("Step 5: Missing values handled.")
    return df_clean

if __name__ == '__main__':
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    file_path = project_root / 'data' / 'GoogleAds_DataAnalytics_Sales_Uncleaned.csv'
    
    # --- Execute the full pipeline ---
    df1 = load_and_standardize_columns(file_path)
    df2 = clean_monetary_columns(df1)
    df3 = standardize_date_column(df2)
    df4 = clean_categorical_columns(df3)
    df_final = handle_missing_values(df4) # The final, fully cleaned DataFrame
    
    # --- Verify the final state ---
    print("\n--- DataFrame Info After Final Cleaning ---")
    df_final.info()
    
    print("\n--- Final Missing Value Counts ---")
    print(df_final.isnull().sum())
    
    print("\n--- Final Value Counts for 'campaign_name' ---")
    print(df_final['campaign_name'].value_counts())