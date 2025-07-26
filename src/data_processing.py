# G:\...\src\data_processing.py
import pandas as pd

def run_full_pipeline(df: pd.DataFrame, training_columns=None) -> pd.DataFrame:
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

    # Monetary
    for col in ['cost', 'sale_amount']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[$,₹]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Date
    if 'ad_date' in df.columns:
        df['ad_date'] = pd.to_datetime(df['ad_date'], format='mixed', errors='coerce')
        df['day_of_week'] = df['ad_date'].dt.dayofweek
        df['month'] = df['ad_date'].dt.month
        df['day_of_month'] = df['ad_date'].dt.day
        df = df.drop(columns=['ad_date'])

    # Categorical Cleaning
    cat_cols = ['campaign_name', 'location', 'device', 'keyword']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].str.lower()

    # Mappings
    if 'campaign_name' in df.columns:
        df['campaign_name'] = df['campaign_name'].str.replace(r'\s+', ' ', regex=True).str.replace('corse', 'course').str.replace('anlytics', 'analytics').str.replace('analytcis', 'analytics')
    if 'location' in df.columns:
        df['location'] = df['location'].str.replace('hyderbad', 'hyderabad').str.replace('hydrebad', 'hyderabad')
    if 'keyword' in df.columns:
        df['keyword'] = df['keyword'].str.replace('analitics', 'analytics').str.replace('anaytics', 'analytics').str.replace('analytic', 'analytics')

    # Missing Values
    if 'conversion_rate' in df.columns:
        df = df.drop(columns=['conversion_rate'])
    
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())

    # One-hot encoding
    cols_to_encode = [col for col in cat_cols if col in df.columns]
    if cols_to_encode:
        df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True, dtype=bool)

    # Align columns
    if training_columns:
        df = df.reindex(columns=training_columns, fill_value=False)
        for col in training_columns:
            if col not in df.columns:
                df[col] = False
    
    if 'ad_id' in df.columns:
        df = df.drop(columns=['ad_id'])
        
    return df