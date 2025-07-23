# G:\...\src\data_processing.py

import pandas as pd
import numpy as np
from pathlib import Path

def run_full_pipeline(df: pd.DataFrame, training_columns=None) -> pd.DataFrame:
    """
    Executes the entire data cleaning and feature engineering pipeline.
    """
    df_processed = df.copy()

    # Step 1: Standardize Column Names
    if any(' ' in col or col.lower() != col for col in df_processed.columns):
        df_processed.columns = df_processed.columns.str.strip().str.lower().str.replace(' ', '_')

    # Step 2: Clean Monetary Columns
    for col in ['cost', 'sale_amount']:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str).str.replace(r'[$,₹]', '', regex=True)
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Step 3: Standardize Date Column
    if 'ad_date' in df_processed.columns:
        df_processed['ad_date'] = pd.to_datetime(df_processed['ad_date'], format='mixed', errors='coerce')

    # Step 4: Clean Categorical Columns
    cols_to_lower = ['campaign_name', 'location', 'device', 'keyword']
    for col in cols_to_lower:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].str.lower()
    
    campaign_mapping = {
        'data anlytics corse': 'data analytics course', 'data analytcis course': 'data analytics course',
        'data analytics corse': 'data analytics course', 'dataanalyticscourse': 'data analytics course'
    }
    location_mapping = { 'hyderbad': 'hyderabad', 'hydrebad': 'hyderabad' }
    keyword_mapping = {
        'data analitics online': 'data analytics online', 'data anaytics training': 'data analytics training',
        'online data analytic': 'online data analytics'
    }
    if 'campaign_name' in df_processed.columns:
        df_processed['campaign_name'] = df_processed['campaign_name'].replace(campaign_mapping)
    if 'location' in df_processed.columns:
        df_processed['location'] = df_processed['location'].replace(location_mapping)
    if 'keyword' in df_processed.columns:
        df_processed['keyword'] = df_processed['keyword'].replace(keyword_mapping)

    # Step 5: Handle Missing Values
    if 'conversion_rate' in df_processed.columns:
        df_processed = df_processed.drop(columns=['conversion_rate'])
    cols_to_fill = ['clicks', 'impressions', 'cost', 'leads', 'conversions', 'sale_amount']
    for col in cols_to_fill:
        if col in df_processed.columns and df_processed[col].isnull().any():
            df_processed[col] = df_processed[col].fillna(0)

    # Step 6: Enforce Float Data Types (Before Engineering)
    dtype_mapping = { 'clicks': 'float64', 'impressions': 'float64', 'cost': 'float64', 'leads': 'float64', 'conversions': 'float64' }
    for col, dtype in dtype_mapping.items():
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(dtype)
    
    # Step 7: Feature Engineering
    if 'ad_date' in df_processed.columns and pd.api.types.is_datetime64_any_dtype(df_processed['ad_date']):
        df_processed['day_of_week'] = df_processed['ad_date'].dt.dayofweek
        df_processed['month'] = df_processed['ad_date'].dt.month
        df_processed['day_of_month'] = df_processed['ad_date'].dt.day
        df_processed = df_processed.drop(columns=['ad_date'])
    if 'ad_id' in df_processed.columns:
        df_processed = df_processed.drop(columns=['ad_id'])
        
    categorical_cols = ['campaign_name', 'location', 'device', 'keyword']
    cols_to_encode = [col for col in categorical_cols if col in df_processed.columns]
    if cols_to_encode:
        # THE FIX: Explicitly set the dtype for one-hot encoding to bool
        df_processed = pd.get_dummies(df_processed, columns=cols_to_encode, drop_first=True, dtype=bool)

    # Step 8: Align Columns with Training Data
    if training_columns:
        # Use False as the fill_value for boolean columns
        df_processed = df_processed.reindex(columns=training_columns, fill_value=False)
        # Ensure all training columns are present and have the correct type
        for col in training_columns:
            if col not in df_processed.columns:
                df_processed[col] = False # Add missing boolean columns
        
    return df_processed