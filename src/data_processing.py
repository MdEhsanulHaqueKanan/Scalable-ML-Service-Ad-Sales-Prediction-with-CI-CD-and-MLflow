# G:\...\src\data_processing.py
import pandas as pd

def run_full_pipeline(df: pd.DataFrame, training_columns: list = None) -> pd.DataFrame:
    df_processed = df.copy()
    df_processed.columns = [col.strip().lower().replace(' ', '_') for col in df_processed.columns]

    # Monetary and numerical columns
    numerical_cols = ['cost', 'sale_amount', 'clicks', 'impressions', 'leads', 'conversions']
    for col in numerical_cols:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(
                df_processed[col].astype(str).str.replace(r'[$,₹]', '', regex=True),
                errors='coerce'
            ).astype(float) # Explicitly cast to float

    # Date features
    if 'ad_date' in df_processed.columns:
        df_processed['ad_date'] = pd.to_datetime(df_processed['ad_date'], format='mixed', errors='coerce')
        df_processed['day_of_week'] = df_processed['ad_date'].dt.dayofweek
        df_processed['month'] = df_processed['ad_date'].dt.month
        df_processed['day_of_month'] = df_processed['ad_date'].dt.day
        df_processed = df_processed.drop(columns=['ad_date'])

    # Categorical cleaning
    cat_cols = ['campaign_name', 'location', 'device', 'keyword']
    for col in cat_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].str.lower().str.strip()
            if col == 'location':
                df_processed[col] = df_processed[col].replace({'hydrebad': 'hyderabad', 'hyderbad': 'hyderabad'})

    # Missing values
    for col in df_processed.columns:
        if df_processed[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df_processed[col]):
                df_processed[col] = df_processed[col].fillna(0) # Fill with 0 for simplicity

    # One-hot encoding
    cols_to_encode = [col for col in cat_cols if col in df_processed.columns]
    if cols_to_encode:
        df_processed = pd.get_dummies(df_processed, columns=cols_to_encode, drop_first=True, dtype=bool)

    # Drop identifiers
    if 'ad_id' in df_processed.columns:
        df_processed = df_processed.drop(columns=['ad_id'])
    if 'conversion_rate' in df_processed.columns:
        df_processed = df_processed.drop(columns=['conversion_rate'])
        
    # Align columns
    if training_columns:
        for col in training_columns:
            if col not in df_processed.columns:
                df_processed[col] = False
        df_processed = df_processed[training_columns]

    return df_processed