import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df):
    """
    Preprocess the customer churn data for modeling.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw customer data
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed data ready for modeling
    dict
        Dictionary containing fitted encoders and scalers
    """
    # Create a copy
    df_processed = df.copy()
    
    # Convert 'Total Charges' to numeric, handling any non-numeric values
    if 'Total Charges' in df_processed.columns:
        df_processed['Total Charges'] = pd.to_numeric(df_processed['Total Charges'], errors='coerce')
    
    # Initialize dictionary to store transformers
    transformers = {}
    
    # Define numerical and categorical columns
    numerical_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges']
    categorical_cols = [
        'Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service',
        'Multiple Lines', 'Internet Service', 'Online Security', 'Online Backup',
        'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies',
        'Contract', 'Paperless Billing', 'Payment Method'
    ]
    
    # Handle numerical features
    numerical_cols = [col for col in numerical_cols if col in df_processed.columns]
    if numerical_cols:
        scaler = StandardScaler()
        # Fill NaN values with mean before scaling
        df_processed[numerical_cols] = df_processed[numerical_cols].fillna(df_processed[numerical_cols].mean())
        df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
        transformers['numerical_scaler'] = scaler
    
    # Handle categorical features
    categorical_cols = [col for col in categorical_cols if col in df_processed.columns]
    for col in categorical_cols:
        if col in df_processed.columns:
            # Convert to string type first to handle mixed types
            df_processed[col] = df_processed[col].astype(str)
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            transformers[f'{col}_encoder'] = le
    
    # Ensure Churn Value exists and is numeric
    if 'Churn Value' in df_processed.columns:
        df_processed['Churn Value'] = pd.to_numeric(df_processed['Churn Value'], errors='coerce')
    elif 'Churn Label' in df_processed.columns:
        df_processed['Churn Value'] = (df_processed['Churn Label'] == 'Yes').astype(int)
    
    return df_processed, transformers

def prepare_features(df):
    """
    Prepare feature matrix X and target variable y
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed data
        
    Returns:
    --------
    numpy.ndarray
        Feature matrix X
    numpy.ndarray
        Target variable y
    list
        Feature names
    """
    # Define features to use - using original column names
    feature_cols = [
        'Tenure Months', 'Monthly Charges', 'Total Charges',
        'Gender', 'Senior Citizen', 'Partner', 'Dependents',
        'Phone Service', 'Multiple Lines', 'Internet Service',
        'Online Security', 'Online Backup', 'Device Protection',
        'Tech Support', 'Streaming TV', 'Streaming Movies',
        'Contract', 'Paperless Billing', 'Payment Method'
    ]
    
    # Remove any columns that don't exist in the dataframe
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].values
    y = df['Churn Value'].values if 'Churn Value' in df.columns else None
    
    return X, y, feature_cols