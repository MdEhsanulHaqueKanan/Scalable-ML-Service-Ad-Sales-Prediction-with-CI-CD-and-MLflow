# G:\...\src\data_processing.py
import pandas as pd

# We only need this one, simple, robust function.
def run_full_pipeline(df: pd.DataFrame, training_columns: list = None) -> pd.DataFrame:
    df_processed = df.copy()
    df_processed.columns = [col.strip().lower().replace(' ', '_') for col in df_processed.columns]
    for col in ['cost', 'sale_amount', 'clicks', 'impressions', 'leads', 'conversions']:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col].astype(str).str.replace(r'[$,₹]', '', regex=True), errors='coerce').astype(float)
    if 'ad_date' in df_processed.columns:
        df_processed['ad_date'] = pd.to_datetime(df_processed['ad_date'], format='mixed', errors='coerce')
        df_processed['day_of_week'] = df_processed['ad_date'].dt.dayofweek
        df_processed['month'] = df_processed['ad_date'].dt.month
        df_processed['day_of_month'] = df_processed['ad_date'].dt.day
        df_processed = df_processed.drop(columns=['ad_date'])
    cat_cols = ['campaign_name', 'location', 'device', 'keyword']
    for col in cat_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].str.lower().str.strip()
    cols_to_encode = [col for col in cat_cols if col in df_processed.columns]
    if cols_to_encode:
        df_processed = pd.get_dummies(df_processed, columns=cols_to_encode, drop_first=True, dtype=bool)
    if 'ad_id' in df_processed.columns:
        df_processed = df_processed.drop(columns=['ad_id'])
    if 'conversion_rate' in df_processed.columns:
        df_processed = df_processed.drop(columns=['conversion_rate'])
    if training_columns:
        for col in training_columns:
            if col not in df_processed.columns:
                df_processed[col] = False
        df_processed = df_processed[training_columns]
    return df_processed