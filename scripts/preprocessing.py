import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    """Preprocess the raw data for analysis and modeling"""
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Define numerical and categorical columns
    numerical_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges']
    categorical_cols = ['Contract', 'Internet Service', 'Online Security',
                       'Online Backup', 'Device Protection', 'Tech Support',
                       'Streaming TV', 'Streaming Movies', 'Payment Method',
                       'Paperless Billing']
    
    # Handle numerical columns
    for col in numerical_cols:
        if col in df.columns:
            # Remove currency symbols and commas if present
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '')
            # Convert to numeric, coerce errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle categorical columns
    for col in categorical_cols:
        if col in df.columns:
            # Fill NaN values with 'Unknown'
            df[col] = df[col].fillna('Unknown')
            # Convert to string type
            df[col] = df[col].astype(str)
    
    # Create Churn Label if it doesn't exist (1 for Yes, 0 for No)
    if 'Churn Label' not in df.columns and 'Churn Value' in df.columns:
        df['Churn Label'] = df['Churn Value'].map({1: 'Yes', 0: 'No'})
    
    return df

def prepare_features(df):
    """Prepare features for modeling"""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Define features to use
    numerical_features = ['Tenure Months', 'Monthly Charges', 'Total Charges']
    categorical_features = ['Contract', 'Internet Service', 'Online Security',
                          'Online Backup', 'Device Protection', 'Tech Support',
                          'Payment Method', 'Paperless Billing']
    
    # Initialize label encoders dictionary
    label_encoders = {}
    
    # Handle categorical features
    for feature in categorical_features:
        if feature in df.columns:
            # Create and fit label encoder
            le = LabelEncoder()
            # Handle missing values before encoding
            df[feature] = df[feature].fillna('Unknown')
            # Fit and transform
            df[feature] = le.fit_transform(df[feature].astype(str))
            # Store the encoder
            label_encoders[feature] = le
    
    # Handle numerical features
    for feature in numerical_features:
        if feature in df.columns:
            # Fill missing values with median
            df[feature] = df[feature].fillna(df[feature].median())
    
    # Select features for modeling
    features = [f for f in numerical_features + categorical_features if f in df.columns]
    X = df[features]
    
    return X

if __name__ == "__main__":
    # Test the functions
    test_df = pd.DataFrame({
        'Tenure Months': [12, 24, 36],
        'Monthly Charges': ['$50.00', '$75.50', '$100.00'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'Churn Value': [1, 0, 0]
    })
    
    processed_df = preprocess_data(test_df)
    features_df = prepare_features(processed_df)
    print("Processed DataFrame:")
    print(processed_df.head())
    print("\nFeatures DataFrame:")
    print(features_df.head())