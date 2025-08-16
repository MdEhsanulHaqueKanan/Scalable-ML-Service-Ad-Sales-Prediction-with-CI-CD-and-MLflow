# G:\...\src\data_processing.py
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# This class will be our entire data processing pipeline
class DataProcessingPipeline(BaseEstimator, TransformerMixin):
    def __init__(self):
        # This will store the column order from the training data
        self.training_columns_ = None

    def fit(self, X, y=None):
        # In fit, we learn the final column structure from the training data
        # We process the data once to determine the final set of columns after one-hot encoding
        processed_df = self._process(X.copy())
        self.training_columns_ = processed_df.columns
        return self

    def transform(self, X, y=None):
        # In transform, we process the new data
        processed_df = self._process(X.copy())
        
        # Align columns to match the training data
        # Add any missing columns that were in the training set (e.g., a dummy variable)
        for col in self.training_columns_:
            if col not in processed_df.columns:
                processed_df[col] = False 
        
        # Ensure the final DataFrame has the exact same columns in the exact same order
        return processed_df[self.training_columns_]

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        # This private method contains our proven, robust cleaning and feature engineering logic
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

        # Clean and convert monetary columns to float
        for col in ['cost', 'clicks', 'impressions', 'leads', 'conversions']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[$,₹]', '', regex=True), errors='coerce')
        
        # Handle dates and create date features
        if 'ad_date' in df.columns:
            df['ad_date'] = pd.to_datetime(df['ad_date'], format='mixed', errors='coerce')
            df['day_of_week'] = df['ad_date'].dt.dayofweek
            df['month'] = df['ad_date'].dt.month
            df['day_of_month'] = df['ad_date'].dt.day
            df = df.drop(columns=['ad_date'])

        # Clean categorical text data
        for col in ['campaign_name', 'location', 'device', 'keyword']:
            if col in df.columns:
                df[col] = df[col].str.lower().str.strip()
                if col == 'location':
                    df[col] = df[col].replace({'hydrebad': 'hyderabad', 'hyderbad': 'hyderabad'})

        # Handle Missing Values in numerical columns
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].isnull().any():
                df[col] = df[col].fillna(0) # Fill with 0 for simplicity

        # One-hot encode the categorical columns that exist
        cols_to_encode = [col for col in ['campaign_name', 'location', 'device', 'keyword'] if col in df.columns]
        if cols_to_encode:
            df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True, dtype=bool)

        # Drop identifier if it exists
        if 'ad_id' in df.columns:
            df = df.drop(columns=['ad_id'])
            
        return df