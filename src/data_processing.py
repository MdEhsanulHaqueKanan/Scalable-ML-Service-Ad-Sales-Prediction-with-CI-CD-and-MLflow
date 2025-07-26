# G:\...\src\data_processing.py
import pandas as pd

def run_full_pipeline(df: pd.DataFrame, training_columns: list = None) -> pd.DataFrame:
    """
    Executes the entire data cleaning and feature engineering pipeline on a given DataFrame.
    """
    # 1. Create a copy to avoid modifying the original DataFrame
    processed_df = df.copy()

    # 2. Standardize column names to snake_case
    processed_df.columns = [col.strip().lower().replace(' ', '_') for col in processed_df.columns]

    # 3. Clean and convert monetary columns
    for col in ['cost', 'sale_amount']:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(
                processed_df[col].astype(str).str.replace(r'[$,₹]', '', regex=True), 
                errors='coerce'
            )

    # 4. Handle dates and create date features
    if 'ad_date' in processed_df.columns:
        processed_df['ad_date'] = pd.to_datetime(processed_df['ad_date'], format='mixed', errors='coerce')
        processed_df['day_of_week'] = processed_df['ad_date'].dt.dayofweek
        processed_df['month'] = processed_df['ad_date'].dt.month
        processed_df['day_of_month'] = processed_df['ad_date'].dt.day
        processed_df = processed_df.drop(columns=['ad_date'])

    # 5. Clean categorical text data
    categorical_cols = ['campaign_name', 'location', 'device', 'keyword']
    for col in categorical_cols:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].str.lower()
            if col == 'location':
                processed_df[col] = processed_df[col].replace({'hydrebad': 'hyderabad', 'hyderbad': 'hyderabad'})

    # 6. Handle Missing Values in numerical columns
    numerical_cols_to_fill = ['clicks', 'impressions', 'cost', 'leads', 'conversions', 'sale_amount']
    for col in numerical_cols_to_fill:
         if col in processed_df.columns and processed_df[col].isnull().any():
            processed_df[col] = processed_df[col].fillna(0)


    # 7. One-hot encode the categorical columns that exist
    cols_to_encode = [col for col in categorical_cols if col in processed_df.columns]
    if cols_to_encode:
        processed_df = pd.get_dummies(processed_df, columns=cols_to_encode, drop_first=True, dtype=bool)

    # 8. Drop identifier and irrelevant columns
    if 'ad_id' in processed_df.columns:
        processed_df = processed_df.drop(columns=['ad_id'])
    if 'conversion_rate' in processed_df.columns:
        processed_df = processed_df.drop(columns=['conversion_rate'])
        
    # 9. Align with training columns if provided (for prediction)
    if training_columns:
        for col in training_columns:
            if col not in processed_df.columns:
                processed_df[col] = False  # Add missing dummy columns as False
        # Ensure the final DataFrame has the exact same columns in the exact same order
        processed_df = processed_df[training_columns]

    return processed_df