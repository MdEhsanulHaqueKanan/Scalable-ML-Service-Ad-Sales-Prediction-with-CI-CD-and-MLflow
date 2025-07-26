# G:\...\src\data_processing.py
import pandas as pd

def run_full_pipeline(df: pd.DataFrame, training_columns=None) -> pd.DataFrame:
    # Make a copy to avoid side effects
    df_processed = df.copy()

    # Standardize column names
    df_processed.columns = [col.strip().lower().replace(' ', '_') for col in df_processed.columns]

    # Clean and convert monetary columns
    for col in ['cost', 'sale_amount']:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col].astype(str).str.replace(r'[$,₹]', '', regex=True), errors='coerce')

    # Handle dates and create date features
    if 'ad_date' in df_processed.columns:
        df_processed['ad_date'] = pd.to_datetime(df_processed['ad_date'], format='mixed', errors='coerce')
        df_processed['day_of_week'] = df_processed['ad_date'].dt.dayofweek
        df_processed['month'] = df_processed['ad_date'].dt.month
        df_processed['day_of_month'] = df_processed['ad_date'].dt.day
        df_processed = df_processed.drop(columns=['ad_date'])

    # Clean categorical text data
    for col in ['campaign_name', 'location', 'device', 'keyword']:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].str.lower()
            if col == 'location':
                df_processed[col] = df_processed[col].replace({'hydrebad': 'hyderabad', 'hyderbad': 'hyderabad'})

    # One-hot encode the categorical columns that exist
    cols_to_encode = [col for col in ['campaign_name', 'location', 'device', 'keyword'] if col in df_processed.columns]
    if cols_to_encode:
        df_processed = pd.get_dummies(df_processed, columns=cols_to_encode, drop_first=True, dtype=bool)

    # Drop identifier and any leftover columns we don't need
    if 'ad_id' in df_processed.columns:
        df_processed = df_processed.drop(columns=['ad_id'])
    if 'conversion_rate' in df_processed.columns:
        df_processed = df_processed.drop(columns=['conversion_rate'])
        
    # Align with training columns if provided
    if training_columns:
        for col in training_columns:
            if col not in df_processed.columns:
                df_processed[col] = False  # Add missing columns as False
        df_processed = df_processed[training_columns] # Ensure exact order and columns

    return df_processed